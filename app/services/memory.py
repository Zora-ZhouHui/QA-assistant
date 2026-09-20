"""会话记忆：可读文件式存储 + 滚动摘要压缩。

存储布局（每个 session 一个目录）：
  data/sessions/<session_id>/
    ├── archive.jsonl    # 全量归档，只追加、永不改写（供历史查询展示给"人"看）
    ├── messages.jsonl   # LLM 工作窗口：近轮原文，压缩时被重写裁剪（给 LLM 看）
    └── summary.md       # 滚动摘要，每次压缩追加一段（便于 git diff 看演变）

压缩视图与全量存储分离：
  压缩只影响送入 LLM 的上下文（messages.jsonl + summary.md），
  全量原文永远留在 archive.jsonl，历史查询读归档而非工作窗口。

为什么用可读文件而不是内存/SQLite？
  - 验证期需要直接打开看压缩是否得当；
  - 可 diff、可手工编辑（改完下一轮 LLM 即读改后版本）；
  - 单机单用户场景，无并发压力，文件 IO 足够。
  跨会话/多 worker/多用户时再升级为 SQLite，上层接口不变。

压缩策略（滚动）：
  近轮原文估算 token 超过 memory_window_tokens 时，从最老的开始取出一批，
  喂给 LLM 压成 ≤300 字摘要，append 到 summary.md，并从 messages.jsonl 删掉那批。
  越老的越压缩、越新的越完整——这是所有记忆方案的本质。
"""
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from app.config import settings
from app.core.llm import get_llm_client

# session_id 白名单：只允许字母数字/_/-，长度 8-128。
# UUID 满足此格式；这同时防住路径穿越（../ 等）。
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")

COMPRESS_PROMPT = """下面是之前对话的片段。请提炼成不超过 300 字的要点，写入长期记忆。要求：
- 保留：用户的核心意图、已确认的结论、尚未解决的问题、关键名词的指代关系。
- 丢弃：寒暄、重复内容、已废弃的中间尝试。
- 用客观陈述，不要对话语气。

对话片段：
{transcript}"""


@dataclass
class SessionMemory:
    """一个会话的内存态快照：近轮原文 + 滚动摘要。"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    running_summary: str = ""


class MemoryService:
    # ---------- 路径与校验 ----------
    @staticmethod
    def _validate(session_id: str) -> None:
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError(f"非法 session_id: {session_id!r}")

    def _dir(self, session_id: str) -> Path:
        d = settings.sessions_dir / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------- 读写 ----------
    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, str]]:
        """逐行读 jsonl，返回消息列表；文件不存在视为空。"""
        messages: List[Dict[str, str]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
        return messages

    def load(self, session_id: str) -> SessionMemory:
        """读出近轮原文与滚动摘要（LLM 工作窗口）。文件不存在视为空会话。"""
        self._validate(session_id)
        d = self._dir(session_id)

        summary_file = d / "summary.md"
        running_summary = summary_file.read_text(encoding="utf-8") if summary_file.exists() else ""
        messages = self._read_jsonl(d / "messages.jsonl")
        return SessionMemory(messages=messages, running_summary=running_summary)

    def load_archive(self, session_id: str) -> List[Dict[str, str]]:
        """读取全量归档（含已被压缩的早期消息），供历史查询展示。

        该机制上线前的旧会话没有 archive.jsonl，退回 messages.jsonl 现存内容；
        若旧会话的早期消息已被压缩删除，则无法找回（当时确实没存）。
        """
        self._validate(session_id)
        d = self._dir(session_id)
        archive = d / "archive.jsonl"
        source = archive if archive.exists() else d / "messages.jsonl"
        return self._read_jsonl(source)

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息：archive.jsonl 全量归档 + messages.jsonl 工作窗口（均 O(1)）。"""
        self._validate(session_id)
        record = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        d = self._dir(session_id)
        with (d / "archive.jsonl").open("a", encoding="utf-8") as f:
            f.write(record + "\n")
        with (d / "messages.jsonl").open("a", encoding="utf-8") as f:
            f.write(record + "\n")

    # ---------- 压缩 ----------
    async def maybe_compress(self, session_id: str) -> None:
        """超阈值时把最老的一批消息压成摘要，从原文中移除。

        放在每轮回答落库后由 asyncio.create_task 后台调用，不阻塞 SSE。
        压缩失败（LLM 异常/空输出）时不动文件，下一轮再试，绝不阻断主流程。
        """
        self._validate(session_id)
        mem = self.load(session_id)
        if not mem.messages:
            return

        # 1) 没超阈值，不动
        if self._estimate_tokens(mem.messages) <= settings.memory_window_tokens:
            return

        # 2) 从最老开始取出，直到剩余回落到 target 以下；至少保留最近 2 条
        remaining = list(mem.messages)
        to_compress: List[Dict[str, str]] = []
        while (
            remaining
            and self._estimate_tokens(remaining) > settings.memory_target_tokens
            and len(remaining) > 2
        ):
            to_compress.append(remaining.pop(0))

        if not to_compress:
            return

        # 3) 调 LLM 压缩（失败则放弃本轮压缩）
        transcript = self._format_messages(to_compress)
        try:
            summary = await self._compress_llm(transcript, mem.running_summary)
        except Exception:
            return
        if not summary:
            return

        # 4) 写回：摘要 append 到 summary.md，messages.jsonl 重写为剩余部分
        self._append_summary(session_id, summary)
        self._rewrite_messages(session_id, remaining)

    async def _compress_llm(self, transcript: str, existing_summary: str) -> str:
        """调用 LLM 把对话片段压成摘要。复用 stream_chat，非流式收集。"""
        messages: List[Dict[str, str]] = []
        if existing_summary:
            messages.append({
                "role": "system",
                "content": f"已有的历史摘要（供参考，新摘要应延续而非重复其中内容）：\n{existing_summary}",
            })
        messages.append({"role": "user", "content": COMPRESS_PROMPT.format(transcript=transcript)})

        parts: List[str] = []
        async for token in get_llm_client().stream_chat(messages):
            parts.append(token)
        return "".join(parts).strip()

    # ---------- 工具 ----------
    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, str]]) -> int:
        # 中文粗估：主流中英 tokenizer 约 1 字 ≈ 0.75 token；偏保守，宁可早压不要爆上下文
        return int(sum(len(m["content"]) for m in messages) * 0.75)

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        lines = []
        for m in messages:
            role = "用户" if m["role"] == "user" else "助手"
            lines.append(f"{role}：{m['content']}")
        return "\n\n".join(lines)

    def _append_summary(self, session_id: str, summary: str) -> None:
        """把新摘要作为一段追加到 summary.md，带时间戳便于 diff 看演变。"""
        sf = self._dir(session_id) / "summary.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        block = f"## 摘要更新 {timestamp}\n\n{summary}\n"
        exists = sf.exists()
        with sf.open("a", encoding="utf-8") as f:
            if exists:
                f.write("\n")  # 段间空行
            f.write(block)

    def _rewrite_messages(self, session_id: str, remaining: List[Dict[str, str]]) -> None:
        """压缩后重写 messages.jsonl，只保留窗口内的近轮原文。"""
        mf = self._dir(session_id) / "messages.jsonl"
        lines = [json.dumps(m, ensure_ascii=False) for m in remaining]
        mf.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # ---------- 会话列表 / 删除 ----------
    def list_sessions(self) -> List[Dict[str, str]]:
        """列出所有会话，按最后活跃时间倒序。

        每个会话返回：session_id / title（首条用户问题前 30 字）/
        message_count / last_active_at（messages.jsonl 的 mtime）。
        没有用户消息的会话标题为"新会话"。
        标题与计数读全量归档（优先 archive.jsonl），压缩不影响它们。
        """
        sessions_dir = settings.sessions_dir
        if not sessions_dir.exists():
            return []

        sessions: List[Dict[str, str]] = []
        for d in sorted(sessions_dir.iterdir()):
            if not d.is_dir():
                continue
            session_id = d.name
            # 非法目录名直接跳过（防御性，正常情况下 mkdir 前已校验）
            if not _SESSION_ID_RE.match(session_id):
                continue

            # 标题/计数读全量：新会话读 archive.jsonl，旧会话退回 messages.jsonl
            active_file = d / "archive.jsonl"
            if not active_file.exists():
                active_file = d / "messages.jsonl"
            messages = self._read_jsonl(active_file)

            # 标题：取第一条用户消息前 30 字
            title = "新会话"
            for m in messages:
                if m.get("role") == "user":
                    title = m["content"][:30]
                    break

            last_active = (
                datetime.fromtimestamp(active_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                if active_file.exists()
                else ""
            )
            sessions.append(
                {
                    "session_id": session_id,
                    "title": title,
                    "message_count": len(messages),
                    "last_active_at": last_active,
                }
            )

        # 按 last_active_at 倒序，新的在前
        sessions.sort(key=lambda s: s["last_active_at"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> None:
        """删除整个会话目录（messages.jsonl + summary.md 一并清除）。"""
        self._validate(session_id)
        d = settings.sessions_dir / session_id
        if d.exists():
            shutil.rmtree(d)


@lru_cache
def get_memory_service() -> MemoryService:
    return MemoryService()

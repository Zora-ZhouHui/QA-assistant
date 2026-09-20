"""Agent 运行时：ReAct 工具循环（从单一 RAG 升级为 Agentic RAG 的核心）。

循环逻辑：
  1. 组装初始消息（提示词 + 记忆 + 问题；KB 资料不再前置注入，由模型自决检索）
  2. 流式请求 LLM（携带可用工具 schema）
       - token 事件 → 直接透传给前端
       - tool_calls → 执行工具 → 结果接续编号回灌 → 进入下一轮请求
  3. 工具执行轮数达到 search_max_rounds 后不再提供工具，
     强制模型基于已有信息收尾作答（防失控、控延迟）。

分层约束：本模块只认识 LLM 抽象（core/llm）与工具注册表（tools/），
不依赖具体模型厂商与搜索引擎——换厂商改 .env，换引擎改 services/search。
"""
import json
import logging
import time
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.agent.prompts import build_system_messages
from app.agent.tools import TOOL_EXECUTORS, get_tools
from app.config import settings
from app.core.llm import get_llm_client
from app.services.memory import SessionMemory

# Agent 执行日志：服务端控制台可见完整决策轨迹（uvicorn 启动即输出）
logger = logging.getLogger("app.agent")


def _format_kb_block(i: int, r: Dict[str, Any]) -> str:
    """单条 KB 资料回灌模型时的格式（与历史 format_contexts 风格一致）。"""
    tag = " · FAQ" if r.get("collection") == "faq" else ""
    return f"【资料{i}】(来源: {r.get('source', '')}{tag})\n{r.get('text', '')}"


def _format_web_block(i: int, r: Dict[str, Any]) -> str:
    """单条网页资料回灌模型时的格式（web_results 带 title 字段）。"""
    return (
        f"[资料{i}] {r.get('title', '')}\n"
        f"来源：{r.get('source', '')}\n"
        f"{r.get('text', '')}"
    )


class AgentService:
    async def run_stream(
        self,
        question: str,
        memory: Optional[SessionMemory] = None,
        web_search_enabled: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """运行一轮问答，产出与 SSE 对应的事件流：
          {"type": "token", "content": str}            —— 回答文本增量
          {"type": "kb_searching", "query": str}       —— 正在检索本地知识库
          {"type": "searching", "query": str}          —— 正在联网搜索
          {"type": "sources", "documents": [...]}      —— KB 检索资料（编号已分配）
          {"type": "web_sources", "documents": [...]}  —— 网页资料（编号接续 KB 之后）
          {"type": "search_failed", ...}               —— 检索失败/空结果
        """
        messages = build_system_messages(question, memory)
        llm = get_llm_client()

        next_index = 1  # 全局编号计数器：KB 与网页资料共用一套编号，按到达顺序接续
        executed_rounds = 0
        llm_calls = 0   # 模型调用总次数（决策 + 生成）

        logger.info(
            "开始处理 | 问题=%r | 联网=%s",
            question[:40], web_search_enabled,
        )

        while True:
            # 轮数用尽后收回所有工具，逼模型基于已有信息收尾
            if executed_rounds < settings.search_max_rounds:
                tools = get_tools(web_search_enabled)
            else:
                tools = []
            tool_calls: List[Dict[str, Any]] = []

            # 每次循环 = 一次模型调用：要么决策调用工具，要么直接生成回答
            llm_calls += 1
            logger.info(
                "模型调用 #%d 开始 | 第%d轮 | 携带工具=%s",
                llm_calls, executed_rounds + 1, bool(tools),
            )
            yield {"type": "llm_start", "call": llm_calls, "round": executed_rounds + 1}

            async for event in llm.stream_chat_with_tools(
                messages, tools=tools or None
            ):
                if event["type"] == "token":
                    yield event
                else:  # tool_calls
                    tool_calls = event["calls"]

            if not tool_calls:
                break  # 模型已给出完整回答

            logger.info(
                "第%d轮 | 模型请求调用工具: %s",
                executed_rounds + 1,
                [c.get("function", {}).get("name") for c in tool_calls],
            )
            executed_rounds += 1
            # OpenAI 协议要求：带 tool_calls 的 assistant 消息先回传到历史
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": tool_calls}
            )

            for call in tool_calls:
                func = call.get("function", {})
                call_id = call.get("id", "")
                name = func.get("name", "")
                arguments = func.get("arguments", "{}")

                executor = TOOL_EXECUTORS.get(name)
                if executor is None:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": f"未知工具：{name}",
                        }
                    )
                    continue

                # 解析 query（kb_search / web_search 均用 query 参数）
                try:
                    query = str(json.loads(arguments or "{}").get("query") or "")
                except json.JSONDecodeError:
                    query = ""

                # 先发"检索中"事件，前端立即展示检索状态
                if name == "kb_search":
                    yield {"type": "kb_searching", "query": query, "round": executed_rounds}
                elif name == "web_search":
                    yield {"type": "searching", "query": query, "round": executed_rounds}

                t0 = time.monotonic()

                result = await executor(arguments)

                # 分流取结果与格式化器：KB 走 sources 事件，网页走 web_sources 事件
                if name == "kb_search":
                    results = result.get("kb_results") or []
                    event_type = "sources"
                    fmt = _format_kb_block
                else:  # web_search
                    results = result.get("web_results") or []
                    event_type = "web_sources"
                    fmt = _format_web_block

                if results:
                    # 编号接续：从 next_index 起分配，KB 与网页共用一套编号
                    start = next_index
                    numbered: List[Dict[str, Any]] = []
                    lines: List[str] = []
                    for i, r in enumerate(results, start=start):
                        item = dict(r)
                        item["index"] = i
                        numbered.append(item)
                        lines.append(fmt(i, r))
                    next_index += len(numbered)
                    yield {"type": event_type, "documents": numbered, "round": executed_rounds}
                    tool_content = (
                        f"检索结果（引用时使用 [资料{start}] 起的编号）：\n\n"
                        + "\n\n".join(lines)
                    )
                else:
                    # 检索失败/空结果：执行器已备好可读的降级说明
                    tool_content = result.get("output", "检索失败")
                    # 轨迹事件：失败/空结果也要在前端轨迹中可见
                    yield {
                        "type": "search_failed",
                        "query": query,
                        "message": tool_content,
                        "round": executed_rounds,
                    }

                logger.info(
                    "工具 %s 完成 | 耗时=%.2fs | 命中=%d条",
                    name, time.monotonic() - t0, len(results),
                )

                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": tool_content}
                )


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService()

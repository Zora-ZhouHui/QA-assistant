"""跨轮图片追问工具：让 Agent 回头"重看"本会话此前上传的截图。

场景：用户第 N 轮上传截图提问，第 N+k 轮追问"图里第 3 行那个报错码是什么"——
此时工作窗口里只有图→文解析文本，解析文本没覆盖的细节答不了。
本工具把会话中最近一次的图片重新交给视觉模型，带着追问定向提取细节。

注册方式特殊：工具依赖 session_id（要知道去哪拿图），不能放进全局 TOOL_EXECUTORS，
由 loop.py 在每次请求时按会话动态注册（见 run_stream 的 session_id 参数）。
"""
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.services.memory import get_memory_service
from app.services.vision import LOOKUP_MAX_CHARS, get_vision_parser

logger = logging.getLogger("app.agent")

VISION_LOOKUP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "vision_lookup",
        "description": (
            "重新查看本会话中用户此前上传的截图，针对追问定向提取图中细节"
            "（报错码、数值、元素文字等）。仅当问题涉及此前截图的图中细节、"
            "且已有图→文解析文本未覆盖时使用；不要用它重复获取已解析过的整体内容。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "想从截图中确认的问题，如「控制台第三行的错误码是什么」",
                }
            },
            "required": ["query"],
        },
    },
}


def find_session_images(session_id: str) -> Optional[List[str]]:
    """从会话归档中找最近一次带图片的用户消息，返回其图片列表；没有则 None。"""
    try:
        archive = get_memory_service().load_archive(session_id)
    except ValueError:
        return None
    for msg in reversed(archive):
        if msg.get("role") == "user" and msg.get("images"):
            return msg["images"]
    return None


def make_executor(session_id: str) -> Callable[[str], Awaitable[Dict[str, Any]]]:
    """生成绑定本会话的 vision_lookup 执行函数（签名与全局工具执行函数一致）。"""
    async def run(arguments: str) -> Dict[str, Any]:
        try:
            query = str(json.loads(arguments or "{}").get("query") or "").strip()
        except json.JSONDecodeError:
            query = ""

        images = find_session_images(session_id)
        if not images:
            return {"output": "本会话没有找到此前上传的截图，无法重看。"}

        parser = get_vision_parser()
        parts: List[str] = []
        try:
            async for chunk in parser.stream_lookup(images, query):
                parts.append(chunk)
        except Exception as e:
            logger.warning("vision_lookup 异常: %s", e)
            return {"output": f"重看截图失败: {e}"}

        text = "".join(parts).strip()
        if not text:
            return {"output": "视觉模型未从截图中提取到与问题相关的内容。"}

        # 回灌上限在这里收口（而非只截断缓存写入）：首次调用与缓存命中回灌一致
        if len(text) > LOOKUP_MAX_CHARS:
            text = text[:LOOKUP_MAX_CHARS] + "\n……（内容过长已截断）"

        # usage 透传给 Agent 循环 → 路由层并入本轮 done 事件的用量/成本汇总
        return {"vision_text": text, "usage": parser.last_usage}

    return run

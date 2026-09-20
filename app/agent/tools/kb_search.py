"""kb_search 工具：本地知识库向量检索。

与 web_search 同构：schema + 异步执行入口。
执行委托给 services/rag 层（RAGService.search），本文件不关心 embedding/向量库细节。

返回约定（供 loop.py 统一处理）：
  kb_results 非空 → loop 给结果接续编号后回灌模型 + 推送前端（sources 事件）；
  kb_results 为空 → output 文本（错误/空结果说明）直接回灌给模型降级。
"""
import json
from typing import Any, Dict

from app.config import settings
from app.services.rag import get_rag_service

KB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "kb_search",
        "description": "检索本地知识库（Agent 面试题、深度学习文档等已入库资料）。"
        "当问题可能涉及本知识库收录的主题时调用；"
        "通用知识、闲聊、与本知识库无关的问题不要调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词，用能体现问题主题的词组，不要用完整句子",
                }
            },
            "required": ["query"],
        },
    },
}


async def run_kb_search(arguments: str) -> Dict[str, Any]:
    try:
        query = str(json.loads(arguments or "{}").get("query") or "").strip()
    except json.JSONDecodeError:
        query = ""
    if not query:
        return {"output": "错误：缺少有效的检索词 query。", "kb_results": []}

    try:
        results = await get_rag_service().search(query, settings.top_k)
    except Exception as e:
        return {
            "output": f"本地知识库检索失败：{e}。请基于你已有的知识回答用户，"
            "并说明未能检索到本地资料。",
            "kb_results": [],
        }

    if not results:
        return {
            "output": f"检索\"{query}\"没有命中本地知识库资料。"
            "可换一组关键词重试，或基于已有知识回答并说明未检索到相关资料。",
            "kb_results": [],
        }
    return {"output": "", "kb_results": results}

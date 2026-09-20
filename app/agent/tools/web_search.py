"""web_search 工具：模型视角的 schema 定义 + 执行入口。

schema 遵循 OpenAI function calling 标准格式（跨厂商通用）；
执行委托给 services/search 层，本文件不关心具体搜索引擎。

返回约定（供 loop.py 统一处理）：
  web_results 非空 → loop 给结果接续编号后回灌模型 + 推送前端；
  web_results 为空 → output 文本（错误/空结果说明）直接回灌给模型降级。
"""
import json
from typing import Any, Dict

from app.services.search import SearchError, get_search_provider

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "联网搜索最新信息。当问题涉及时效信息（最新版本、新闻、价格、政策等）"
        "或知识库资料不足以回答时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，用简洁的词组组合，不要用完整句子",
                }
            },
            "required": ["query"],
        },
    },
}


async def run_web_search(arguments: str) -> Dict[str, Any]:
    try:
        query = str(json.loads(arguments or "{}").get("query") or "").strip()
    except json.JSONDecodeError:
        query = ""
    if not query:
        return {"output": "错误：缺少有效的搜索词 query。", "web_results": []}

    try:
        results = await get_search_provider().search(query)
    except SearchError as e:
        return {
            "output": f"联网搜索失败：{e}。请基于你已有的知识回答用户，"
            "并明确说明未能联网核实、信息可能不是最新的。",
            "web_results": [],
        }

    if not results:
        return {
            "output": f"搜索\"{query}\"没有返回结果。可换一组关键词重试，"
            "或基于已有知识回答并说明不确定性。",
            "web_results": [],
        }
    return {"output": "", "web_results": results}

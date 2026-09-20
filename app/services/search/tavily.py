"""Tavily 适配器：专为 AI agent 设计的搜索 API（免费 1000 次/月）。

接口：POST https://api.tavily.com/search，Bearer 鉴权，
返回结构化结果（标题/URL/清洗过的正文/相关度），无需自己抓网页。
"""
from typing import Any, Dict, List

import httpx

from app.config import settings

from .base import SearchError, SearchProvider


class TavilyProvider(SearchProvider):
    API_URL = "https://api.tavily.com/search"

    async def search(self, query: str) -> List[Dict[str, Any]]:
        if not settings.tavily_api_key:
            raise SearchError("未配置 TAVILY_API_KEY（在 .env 中填入后重启服务）")

        payload: Dict[str, Any] = {
            "query": query,
            "search_depth": settings.search_depth,
            "max_results": settings.search_max_results,
            "include_answer": False,  # 摘要由我们自己的 LLM 生成，不花这笔钱
        }
        if settings.search_country:
            payload["country"] = settings.search_country

        try:
            async with httpx.AsyncClient(timeout=settings.search_timeout) as client:
                resp = await client.post(
                    self.API_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            # 401/432：key 无效或配额用尽；429：限流
            raise SearchError(f"Tavily 返回 HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise SearchError(f"请求 Tavily 失败: {type(e).__name__}") from e

        return [
            {
                "text": (r.get("content") or "").strip(),
                "source": r.get("url") or "",
                "title": (r.get("title") or "").strip(),
                "score": float(r.get("score") or 0.0),
                "collection": "web",
            }
            for r in data.get("results", [])
            if r.get("url")
        ]

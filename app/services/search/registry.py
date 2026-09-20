"""按 SEARCH_PROVIDER 配置取搜索引擎实例。

扩展方式（与 faq_sources 的 registry 一致）：实现一个 SearchProvider
子类，在这里加一个分支即可，上层（agent 工具）零改动。
"""
from functools import lru_cache

from app.config import settings

from .base import SearchError, SearchProvider
from .tavily import TavilyProvider


@lru_cache
def get_search_provider() -> SearchProvider:
    name = (settings.search_provider or "tavily").strip().lower()
    if name == "tavily":
        return TavilyProvider()
    raise SearchError(f"未知搜索引擎: {name}（当前支持 tavily）")

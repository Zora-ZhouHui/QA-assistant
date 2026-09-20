"""搜索通道包：对外只暴露抽象基类、错误类型和工厂函数。"""
from .base import SearchError, SearchProvider
from .registry import get_search_provider

__all__ = ["SearchError", "SearchProvider", "get_search_provider"]

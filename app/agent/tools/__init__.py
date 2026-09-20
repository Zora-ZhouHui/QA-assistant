"""Agent 工具注册表。

新增工具三步：写执行函数 + 写 schema + 在下面两处注册。
执行函数统一签名：async (arguments_json: str) -> {"output"?: str, "web_results"?: list, "kb_results"?: list}
"""
from typing import Any, Awaitable, Callable, Dict, List

from .kb_search import KB_SEARCH_SCHEMA, run_kb_search
from .web_search import WEB_SEARCH_SCHEMA, run_web_search

# 工具名 → 异步执行函数
TOOL_EXECUTORS: Dict[str, Callable[[str], Awaitable[Dict[str, Any]]]] = {
    "web_search": run_web_search,
    "kb_search": run_kb_search,
}


def get_tools(web_search_enabled: bool = True) -> List[Dict[str, Any]]:
    """返回当前可用的工具 schema 列表（OpenAI function calling 标准格式）。

    KB 检索始终可用（与联网开关解耦——关掉联网时仍可查本地库）；
    web_search 受 web_search_enabled 控制。
    """
    tools: List[Dict[str, Any]] = [KB_SEARCH_SCHEMA]
    if web_search_enabled:
        tools.append(WEB_SEARCH_SCHEMA)
    return tools

"""Agent 运行时包：ReAct 工具循环 + 提示词 + 工具注册表。"""
from .loop import AgentService, get_agent_service

__all__ = ["AgentService", "get_agent_service"]

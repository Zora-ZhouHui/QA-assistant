"""跨层数据契约（依赖倒置的"中间层"）。

为什么单独放在 core/ 而不是 services/：
  这些数据形状同时被"编排层(agent)"和"业务服务层(services)"使用，
  但两者之间不应互相依赖。把它放在下层的 core 后：
    agent   → core.contracts   （上层依赖下层契约）
    services → core.contracts  （服务层实现契约）
  两个方向都只向下，编排层不再反向 import 任何 services 具体实现。

这里只放"纯数据形状"，不含行为——实现仍在各自的 services 模块。
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SessionMemory:
    """一个会话的内存态快照：近轮原文 + 滚动摘要。

    由 services/memory.py 的 MemoryService.load() 产出，
    被 agent/prompts.py 组装进 LLM 消息、被 agent/loop.py 作为入参传递。
    """
    messages: List[Dict[str, str]] = field(default_factory=list)
    running_summary: str = ""

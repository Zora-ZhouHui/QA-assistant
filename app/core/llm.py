"""LLM 大模型客户端。

走 OpenAI 兼容接口（当前接 DeepSeek deepseek-chat），直接用官方 openai SDK，
通过 base_url + api_key 切换厂商；换模型只改 .env，无需改代码。

Agent 支持：
  stream_chat_with_tools 是标准 OpenAI function calling 协议的流式实现，
  只用 model/messages/tools/stream/temperature 等跨厂商通用字段，
  不依赖任何厂商私有的联网/知识库工具，因此可随 .env 无缝换厂商。
"""
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.config import settings


class LLMClient:
    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not settings.llm_api_key:
                raise RuntimeError(
                    "未配置 LLM_API_KEY：请复制 .env.example 为 .env 并填入你的密钥"
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
        return self._client

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """纯文本流式对话（不带工具），逐 token yield 文本增量。

        用于后台摘要压缩（memory.py）等"只需文本、不涉及 function calling"
        的内部调用；本质是 stream_chat_with_tools 在 tools=None 时的薄封装，
        避免上层直接消费事件结构。
        """
        async for event in self.stream_chat_with_tools(
            messages, tools=None
        ):
            if event["type"] == "token":
                yield event["content"]

    async def stream_chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式对话，产出事件流：
          {"type": "token", "content": str}        —— 回答文本增量
          {"type": "tool_calls", "calls": [...]}   —— 流结束时模型请求调用工具

        tools 为 None 时等价纯文本流，永远不会产生 tool_calls 事件。

        流式协议下 tool_calls 是分片到达的（按 index 增量拼 arguments），
        这里负责拼装成完整的标准结构，调用方（agent loop）无需感知分片细节。
        """
        client = self._ensure_client()
        kwargs: Dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.3,  # 知识问答场景调低温度，减少发挥
        }
        if tools:
            kwargs["tools"] = tools
        response = await client.chat.completions.create(**kwargs)

        # 流式 tool_calls 分片拼装：index → {id, name, arguments 增量累加}
        tool_acc: Dict[int, Dict[str, str]] = {}
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                yield {"type": "token", "content": delta.content}
            for tc in delta.tool_calls or []:
                slot = tool_acc.setdefault(
                    tc.index, {"id": "", "name": "", "arguments": ""}
                )
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

        # 以"是否真的收到过分片"为准（部分兼容层的 finish_reason 不规范）
        if tool_acc:
            calls = [
                {
                    "id": slot["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": slot["name"],
                        "arguments": slot["arguments"] or "{}",
                    },
                }
                for idx, slot in sorted(tool_acc.items())
            ]
            yield {"type": "tool_calls", "calls": calls}


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()

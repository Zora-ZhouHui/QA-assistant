"""LLM 大模型客户端。

走 OpenAI 兼容接口（当前接阿里云百炼 qwen3.7-plus），直接用官方 openai SDK，
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
        # 按 (base_url, api_key) 缓存多个 client，支持文本/视觉模型走不同厂商端点
        self._clients: Dict[tuple, Any] = {}

    def _ensure_client(self, base_url: str = "", api_key: str = ""):
        base_url = base_url or settings.llm_base_url
        api_key = api_key or settings.llm_api_key
        if not api_key:
            raise RuntimeError(
                "未配置 LLM_API_KEY：请复制 .env.example 为 .env 并填入你的密钥"
            )
        key = (base_url, api_key)
        if key not in self._clients:
            from openai import AsyncOpenAI

            self._clients[key] = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        return self._clients[key]

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
        *,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式对话，产出事件流：
          {"type": "token", "content": str}        —— 回答文本增量
          {"type": "tool_calls", "calls": [...]}   —— 流结束时模型请求调用工具
          {"type": "usage", "usage": {...}, "model": str} —— 本次调用的 token 用量

        tools 为 None 时等价纯文本流，永远不会产生 tool_calls 事件。

        流式协议下 tool_calls 是分片到达的（按 index 增量拼 arguments），
        这里负责拼装成完整的标准结构，调用方（agent loop）无需感知分片细节。

        token 计量：stream_options.include_usage 让厂商在流的最后一个 chunk
        返回精确 usage（账单口径）；该 chunk 的 choices 为空，须在分片循环里
        单独捕获，流结束后以 usage 事件产出。
        """
        client = self._ensure_client(base_url, api_key)
        model_id = model or settings.llm_model
        kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.3,  # 知识问答场景调低温度，减少发挥
            # 百炼系模型（qwen3.7-plus 等）默认开启思考模式：思考内容走 reasoning_content
            # 字段（本客户端只消费 content，会静默丢弃），且思考 token 按输出计费。
            # 显式关闭以降低延迟与成本；该参数非 OpenAI 标准，必须经 extra_body 传入。
            "extra_body": {"enable_thinking": False},
        }
        if tools:
            kwargs["tools"] = tools
        response = await client.chat.completions.create(**kwargs)

        # 流式 tool_calls 分片拼装：index → {id, name, arguments 增量累加}
        tool_acc: Dict[int, Dict[str, str]] = {}
        usage_seen: Optional[Dict[str, int]] = None
        async for chunk in response:
            # usage 挂在末个 chunk 上（choices 为空），先捕获再走 choices 判断
            if getattr(chunk, "usage", None):
                u = chunk.usage
                usage_seen = {
                    "prompt_tokens": u.prompt_tokens or 0,
                    "completion_tokens": u.completion_tokens or 0,
                    "total_tokens": u.total_tokens or 0,
                }
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
        if usage_seen:
            yield {"type": "usage", "usage": usage_seen, "model": model_id}
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

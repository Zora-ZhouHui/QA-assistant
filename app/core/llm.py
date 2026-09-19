"""LLM 大模型客户端。

走 OpenAI 兼容接口（当前接 GLM-4.7-Flash），直接用官方 openai SDK，
通过 base_url + api_key 切换厂商；换模型只改 .env，无需改代码。
"""
from functools import lru_cache
from typing import AsyncGenerator, Dict, List

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
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """流式对话：逐 token 返回模型输出，前端可以边生成边显示。"""
        client = self._ensure_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=True,
            temperature=0.3,  # 知识问答场景调低温度，减少发挥
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()

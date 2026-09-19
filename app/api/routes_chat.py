"""问答接口：SSE（Server-Sent Events）流式返回。

为什么用 SSE？
  大模型是逐 token 生成的，流式返回可以让用户边生成边看，
  不用等整段回答完。事件类型：
    sources —— 检索到的参考资料（先发给前端展示）
    token   —— 回答的一个片段
    error   —— 出错信息
    done    —— 回答结束

记忆接入：
  请求带 session_id，回答前加载该会话的历史（近轮原文 + 摘要）拼进上下文，
  回答落库后异步触发压缩。session_id 由前端生成（UUID）并存 localStorage。
"""
import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services.memory import get_memory_service
from app.services.rag import get_rag_service

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.top_k, ge=1, le=10)
    session_id: str = Field(..., min_length=8, max_length=128)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def generate():
        service = get_rag_service()
        mem_service = get_memory_service()

        # 加载该会话的记忆（近轮原文 + 滚动摘要）；非法 session_id 在此抛错
        try:
            memory = mem_service.load(req.session_id)
        except ValueError as e:
            yield _sse({"type": "error", "message": f"会话标识非法: {e}"})
            return

        try:
            contexts = await service.search(req.question, req.top_k)
        except Exception as e:
            yield _sse({"type": "error", "message": f"检索失败: {e}"})
            return

        yield _sse(
            {
                "type": "sources",
                "documents": [
                    {
                        "source": c["source"],
                        "score": c["score"],
                        # 区分文档切片 vs FAQ；FAQ 额外带 question 文本
                        "collection": c.get("collection", "docs"),
                        "question": c.get("question"),
                        # 片段正文：docs 是切片，faq 是"问：…\n答：…"
                        "text": c.get("text", ""),
                    }
                    for c in contexts
                ],
            }
        )

        # 流式生成，同时收集完整回答用于落库
        full_answer: list = []
        try:
            async for token in service.answer_stream(req.question, contexts, memory):
                full_answer.append(token)
                yield _sse({"type": "token", "content": token})
        except Exception as e:
            yield _sse({"type": "error", "message": f"生成失败: {e}"})
            return

        # 本轮落库：用户问题 + 助手回答
        mem_service.append(req.session_id, "user", req.question)
        mem_service.append(req.session_id, "assistant", "".join(full_answer))

        # 后台触发压缩（超阈值才真正压缩），不阻塞本次 SSE 返回
        asyncio.create_task(mem_service.maybe_compress(req.session_id))

        yield _sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")

"""问答接口：SSE（Server-Sent Events）流式返回。

为什么用 SSE？
  大模型是逐 token 生成的，流式返回可以让用户边生成边看，
  不用等整段回答完。事件类型：
    kb_searching —— Agent 正在检索本地知识库（含检索词）
    sources      —— 知识库检索到的参考资料（编号已分配）
    searching    —— Agent 正在联网搜索（含搜索词）
    web_sources  —— 联网搜到的网页资料（编号接续 sources 之后）
    token        —— 回答的一个片段
    error        —— 出错信息
    done         —— 回答结束

编排职责说明：
  本模块只做"加载记忆 → 转发 Agent 事件流 → 落库"的粘合，
  决策（直答/检索本地库/联网）与工具循环全部在 app/agent/loop.py。

记忆接入：
  请求带 session_id，回答前加载该会话的历史（近轮原文 + 摘要）拼进上下文，
  回答落库后异步触发压缩。session_id 由前端生成（UUID）并存 localStorage。
"""
import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent import get_agent_service
from app.config import settings
from app.services.memory import get_memory_service

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.top_k, ge=1, le=10)
    session_id: str = Field(..., min_length=8, max_length=128)
    # 是否允许 Agent 自动联网搜索（前端开关，默认值取配置）
    web_search_enabled: bool = Field(default=settings.web_search_default)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def generate():
        agent = get_agent_service()
        mem_service = get_memory_service()

        # 加载该会话的记忆（近轮原文 + 滚动摘要）；非法 session_id 在此抛错
        try:
            memory = mem_service.load(req.session_id)
        except ValueError as e:
            yield _sse({"type": "error", "message": f"会话标识非法: {e}"})
            return

        # 先落一条"新一轮提问"边界事件：回放时据此把同一 session 内多个问题
        # 的轨迹切分开，否则两个问题的 llm_start 会连成一片、前面的展示不出来。
        mem_service.append_trace(
            req.session_id, {"type": "question", "question": req.question}
        )

        # Agent 工具循环：模型自决检索本地库（kb_search）/ 联网（web_search）/ 直答；
        # KB 检索不再固定前置，无关问题直接作答省掉向量检索等待。收集完整回答用于落库。
        # 轨迹落库辅助：观测类事件（决策/检索轨迹）在透传前端的同时写入 trace.jsonl，
        # 供刷新 / 切换会话后回放；token 是业务内容，回答原文已由 archive 持久化。
        def emit(payload: dict) -> str:
            mem_service.append_trace(req.session_id, payload)
            return _sse(payload)

        full_answer: list = []
        try:
            async for event in agent.run_stream(
                req.question, memory, req.web_search_enabled
            ):
                if event["type"] == "token":
                    full_answer.append(event["content"])
                    yield _sse({"type": "token", "content": event["content"]})
                elif event["type"] == "llm_start":
                    yield emit({"type": "llm_start", "call": event["call"]})
                elif event["type"] == "kb_searching":
                    yield emit({"type": "kb_searching", "query": event["query"]})
                elif event["type"] == "searching":
                    yield emit({"type": "searching", "query": event["query"]})
                elif event["type"] == "sources":
                    yield emit({"type": "sources", "documents": event["documents"]})
                elif event["type"] == "web_sources":
                    yield emit({"type": "web_sources", "documents": event["documents"]})
                elif event["type"] == "search_failed":
                    yield emit(
                        {
                            "type": "search_failed",
                            "query": event["query"],
                            "message": event["message"],
                        }
                    )
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

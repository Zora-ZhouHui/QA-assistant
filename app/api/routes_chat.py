"""问答接口：SSE（Server-Sent Events）流式返回。

为什么用 SSE？
  大模型是逐 token 生成的，流式返回可以让用户边生成边看，
  不用等整段回答完。事件类型：
    vision_start —— 开始用多模态模型解析截图（含场景）
    vision_token —— 图片解析文本的一个片段（只透传前端，不落轨迹）
    vision_done  —— 图片解析完成（含接入 RAG 的完整问题文本）
    kb_searching —— Agent 正在检索本地知识库（含检索词）
    sources      —— 知识库检索到的参考资料（编号已分配）
    searching    —— Agent 正在联网搜索（含搜索词）
    web_sources  —— 联网搜到的网页资料（编号接续 sources 之后）
    token        —— 回答的一个片段
    error        —— 出错信息
    done         —— 回答结束

两条入口（文字入口完全不变，检索逻辑零改动）：
  1) 文字问答：问题直接进入 app/agent/loop.py 的 Agent 工具循环。
  2) 截图问答（新增）：上传图片 + 选定场景 → 多模态模型按场景指令解析出
     纯文本 → 该文本作为问题接入同一套 kb_search + web_search 工具循环。
     多模态只做"图 → 文"，决策（直答/检索本地库/联网）仍全部在 loop.py。

编排职责说明：
  本模块只做"加载记忆 →（截图问答时）多模态解析 → 转发 Agent 事件流 → 落库"的粘合。

记忆接入：
  请求带 session_id，回答前加载该会话的历史（近轮原文 + 摘要）拼进上下文，
  回答落库后异步触发压缩。session_id 由前端生成（UUID）并存 localStorage。
"""
import asyncio
import json
from typing import List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent import get_agent_service
from app.config import settings
from app.services.memory import get_memory_service
from app.services.vision import SCENES, get_vision_parser

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    # 文字入口必填；截图入口可只带图片（scene 指定场景），question 为可选的补充说明
    question: str = Field(default="")
    top_k: int = Field(default=settings.top_k, ge=1, le=10)
    session_id: str = Field(..., min_length=8, max_length=128)
    # 是否允许 Agent 自动联网搜索（前端开关，默认值取配置）
    web_search_enabled: bool = Field(default=settings.web_search_default)
    # 可选的多模态附件：图片 base64 dataURL 列表
    images: List[str] = Field(default_factory=list)
    # 截图场景 key（console_error / layout_misalign / design_compare）；带图请求必填
    scene: str = Field(default="")


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def generate():
        agent = get_agent_service()
        mem_service = get_memory_service()

        images = req.images or []
        scene = (req.scene or "").strip()
        question = (req.question or "").strip()

        # 图片校验（快速失败，不落任何东西）：仅接受 dataURL，且控制单张体积
        for img in images:
            if not (img.startswith("data:image/") and ";base64," in img):
                yield _sse({"type": "error", "message": "图片格式非法：仅支持 data:image/...;base64"})
                return
            if len(img) > settings.max_image_bytes:
                yield _sse({"type": "error", "message": "图片过大，请压缩后重试"})
                return

        if not question and not images:
            yield _sse({"type": "error", "message": "请输入问题或上传截图"})
            return

        # 截图问答前置校验：场景必须合法（视觉模型是否配置由 VisionParser 在调用时校验）
        if images:
            if scene not in SCENES:
                yield _sse(
                    {
                        "type": "error",
                        "message": "请先选择截图场景（控制台报错 / 布局错位 / 设计稿对比）",
                    }
                )
                return

        # 加载该会话的记忆（近轮原文 + 滚动摘要）；非法 session_id 在此抛错
        try:
            memory = mem_service.load(req.session_id)
        except ValueError as e:
            yield _sse({"type": "error", "message": f"会话标识非法: {e}"})
            return

        # 轨迹落库辅助：观测类事件（决策/解析/检索轨迹）在透传前端的同时写入 trace.jsonl，
        # 供刷新 / 切换会话后回放；token 是业务内容，回答原文已由 archive 持久化。
        def emit(payload: dict) -> str:
            mem_service.append_trace(req.session_id, payload)
            return _sse(payload)

        # 截图问答：VisionParser 把图片按场景指令流式解析成纯文本（图→文）。
        # 解析完成后把文本作为问题接入下面的 Agent 工具循环，不改动任何检索逻辑。
        if images:
            spec = SCENES[scene]
            yield emit({"type": "vision_start", "scene": scene, "label": spec.label})

            parser = get_vision_parser()
            parts: List[str] = []
            try:
                async for chunk in parser.stream_parse(images, scene, note=question):
                    parts.append(chunk)
                    yield _sse({"type": "vision_token", "content": chunk})
            except RuntimeError as e:
                yield _sse({"type": "error", "message": str(e)})
                return
            except ValueError as e:
                yield _sse({"type": "error", "message": str(e)})
                return
            except Exception as e:
                yield _sse({"type": "error", "message": f"图片解析失败: {e}"})
                return

            analysis = "".join(parts).strip()
            if not analysis:
                yield _sse({"type": "error", "message": "图片解析未返回内容，请更换截图后重试"})
                return

            # 直接作答场景（如设计稿对比）：视觉解析结果即最终答案，跳过 Agent/RAG。
            # 其余场景：解析文本 + 按场景的追问句（驱动模型调用检索工具），补充说明可选追加。
            direct_answer = spec.direct_answer
            if direct_answer:
                rag_question = analysis
            else:
                rag_question = f"{analysis}\n\n{spec.followup}"
                if question:
                    rag_question = f"{rag_question}\n\n补充说明：{question}"

            yield emit(
                {
                    "type": "vision_done",
                    "scene": scene,
                    "label": spec.label,
                    "question": rag_question,
                }
            )
            # 日志/回放里的"提问"节点用场景名（解析全文已在用户消息中展示，避免重复堆叠）
            log_question = f"截图场景：{spec.label}"
        else:
            rag_question = question
            log_question = question
            direct_answer = False

        # 先落一条"新一轮提问"边界事件：回放时据此把同一 session 内多个问题
        # 的轨迹切分开，否则两个问题的 llm_start 会连成一片、前面的展示不出来。
        mem_service.append_trace(
            req.session_id, {"type": "question", "question": log_question}
        )

        # Agent 工具循环（截图问答与文字问答共用同一套逻辑）：模型自决检索本地库
        # （kb_search）/ 联网（web_search）/ 直答；KB 检索不再固定前置，无关问题
        # 直接作答省掉向量检索等待。收集完整回答用于落库。
        # 直接作答场景（如设计稿对比）跳过该循环：视觉对比文本即最终回答。
        full_answer: list = []
        if direct_answer:
            full_answer.append(rag_question)
            yield _sse({"type": "token", "content": rag_question})
        else:
            try:
                async for event in agent.run_stream(
                    rag_question, memory, req.web_search_enabled
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

        # 本轮落库：用户消息 + 助手回答。截图问答时归档只存用户原文（可能为空，
        # 图片随归档展示）；解析文本是模型产物，只进 LLM 工作窗口提供语境。
        if images:
            mem_service.append(
                req.session_id,
                "user",
                question,
                images=images or None,
                window_content=rag_question,
            )
        else:
            mem_service.append(req.session_id, "user", rag_question)
        mem_service.append(req.session_id, "assistant", "".join(full_answer))

        # 后台触发压缩（超阈值才真正压缩），不阻塞本次 SSE 返回
        asyncio.create_task(mem_service.maybe_compress(req.session_id))

        yield _sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")

"""会话管理接口：列出 / 创建 / 删除 / 读取历史消息。

会话记忆的存储与校验复用 app.services.memory.MemoryService：
  - 每个 session 一个目录 data/sessions/<session_id>/（messages.jsonl + summary.md）
  - session_id 由前端生成 UUID 传入，后端只做白名单校验（防路径穿越）
"""
import uuid

from fastapi import APIRouter, HTTPException

from app.services.memory import get_memory_service

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_sessions() -> dict:
    """列出所有会话（按最后活跃时间倒序）。"""
    sessions = get_memory_service().list_sessions()
    return {"sessions": sessions}


@router.post("")
def create_session() -> dict:
    """创建新会话，返回 session_id。

    实际落盘延迟到首次提问时（MemoryService._dir 会 mkdir），
    这里只生成一个合法 UUID 并返回，前端立刻切换为当前会话。
    """
    session_id = uuid.uuid4().hex
    return {"session_id": session_id}


@router.delete("/{session_id}")
def delete_session(session_id: str) -> dict:
    """删除指定会话及其全部消息。"""
    try:
        get_memory_service().delete_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@router.get("/{session_id}/messages")
def get_session_messages(session_id: str) -> dict:
    """读取指定会话的全部历史消息（正序：旧→新）。

    读全量归档而非 LLM 工作窗口，被压缩的早期消息也能查到。
    """
    try:
        messages = get_memory_service().load_archive(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"messages": messages}


@router.get("/{session_id}/trace")
def get_session_trace(session_id: str) -> dict:
    """读取指定会话的执行轨迹（决策/检索过程），供历史会话回放。

    与 /messages 读取对话原文互补：这里返回 llm_start/searching/sources 等
    观测事件的正序列表，前端据此还原右侧执行日志面板。
    """
    try:
        events = get_memory_service().load_trace(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"events": events}

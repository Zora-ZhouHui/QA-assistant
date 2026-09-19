"""RAG 核心流程编排。

一条完整的问答链路：
  入库：data/ 目录下的知识文件 → 解析纯文本 → 切片 → bge 向量化 → 存入 Chroma
        （服务启动时由 app.services.indexer 自动完成，无需手动上传）
  问答：问题向量化 → Chroma 相似度检索 top_k → 切片拼进 prompt → GLM-4.7-Flash 流式生成
"""
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.core.embeddings import get_embedder
from app.core.llm import get_llm_client
from app.core.vectorstore import get_vector_store
from app.services.document_loader import load_text
from app.services.embedding_snapshot import get_snapshot
from app.services.memory import SessionMemory
from app.services.splitter import split_text

SYSTEM_PROMPT = """你是一个知识问答助手。请优先根据"参考资料"回答用户的问题，并可结合"对话历史"理解用户的意图与指代。

要求：
1. 优先使用参考资料中的信息；参考资料未覆盖的部分，可结合对话历史中已确认的结论作答，但需区分"资料所述"与"基于历史对话的推断"。
2. 如果参考资料与对话历史均不足以回答问题，直接说"根据现有资料，无法回答该问题"。
3. 回答简洁、准确，并在末尾标注引用了哪些资料编号（如 [资料1]）。
4. 当用户问题涉及之前的对话内容（如"刚才说的""那个方法"），务必参考对话历史摘要与近轮原文理解指代。"""


def build_messages(
    question: str,
    contexts: List[Dict[str, Any]],
    memory: Optional[SessionMemory] = None,
) -> List[Dict[str, str]]:
    """组装三层上下文：参考资料(系统) + 历史摘要(系统) + 近轮原文 + 当前问题。

    层次从上到下：越稳定/越压缩的越靠上（资料、摘要），越新鲜/越完整的越靠下（近轮、问题）。
    memory 为空会话时 messages 为空、摘要为空，等价于无记忆（兼容首次提问）。
    context 带 collection 字段：docs 项是文档切片，faq 项是 Q&A 对（text 已含问/答）。
    在来源标注后附 " · FAQ" 让模型清楚引用类型。
    """
    if contexts:
        blocks = []
        for i, ctx in enumerate(contexts, start=1):
            tag = " · FAQ" if ctx.get("collection") == "faq" else ""
            blocks.append(f"【资料{i}】(来源: {ctx['source']}{tag})\n{ctx['text']}")
        context_text = "\n\n".join(blocks)
    else:
        context_text = "（知识库为空，没有检索到任何资料）"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n参考资料：\n{context_text}"},
    ]
    if memory and memory.running_summary:
        messages.append(
            {"role": "system", "content": f"对话历史摘要：\n{memory.running_summary}"}
        )
    if memory and memory.messages:
        messages.extend(memory.messages)
    messages.append({"role": "user", "content": question})
    return messages


class RAGService:
    # ---------- 入库 ----------
    def ingest_file(self, file_path: Path, source: str, doc_id: str) -> Dict[str, Any]:
        """解析 + 切片 + 向量化 + 写库。

        doc_id 由调用方给定（同一文件路径对应稳定 id），
        这样文件更新时可以先删旧向量再重新写入。
        返回 doc_id 和切片数。
        """
        text = load_text(file_path)
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("文档内容为空，无法入库")

        # 分块之后、向量化之前：把这批切片落盘成快照，供人工核对
        get_snapshot().replace_doc(doc_id, source, chunks)

        embeddings = get_embedder().embed_documents(chunks)
        get_vector_store().add(doc_id, chunks, embeddings, source=source)
        return {"doc_id": doc_id, "chunk_count": len(chunks)}

    def ingest_faq(
        self,
        faq_id: str,
        question: str,
        answer: str,
        source_url: str,
        section: str,
    ) -> Dict[str, Any]:
        """FAQ 入库：只对 question 做 embedding，answer 仅作为 metadata 存储。

        注意：用 embed_documents 而非 embed_query——FAQ 的 question 在这里是被检索的
        "文档侧"，不应加 query 指令前缀；用户输入才是 query 端。
        """
        # 向量化之前：FAQ 只有 question 进 embedder，快照也只记录 question
        get_snapshot().replace_faq(faq_id, question, section, source_url)

        question_embedding = get_embedder().embed_documents([question])[0]
        get_vector_store().add_faq(
            faq_id=faq_id,
            question=question,
            answer=answer,
            source_url=source_url,
            section=section,
            question_embedding=question_embedding,
        )
        return {"faq_id": faq_id}

    # ---------- 检索 ----------
    async def search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        # embedding 计算是 CPU 密集的同步操作，放到线程池避免阻塞事件循环
        embedder = get_embedder()
        query_vector = await run_in_threadpool(embedder.embed_query, question)
        store = get_vector_store()
        # 综合搜索 docs + faq，按 cosine 相似度全局排序后返回 top_k
        return await run_in_threadpool(store.query_all, query_vector, top_k)

    # ---------- 生成 ----------
    async def answer_stream(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        memory: Optional[SessionMemory] = None,
    ) -> AsyncGenerator[str, None]:
        messages = build_messages(question, contexts, memory)
        async for token in get_llm_client().stream_chat(messages):
            yield token


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService()

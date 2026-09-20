"""RAG 检索服务（Agent 的"知识库感知"层）。

一条完整的问答链路：
  入库：data/ 目录下的知识文件 → 解析纯文本 → 切片 → bge 向量化 → 存入 Chroma
        （服务启动时由 app.services.indexer 自动完成，无需手动上传）
  问答：问题向量化 → Chroma 相似度检索 top_k → 结果交给 Agent
        （消息组装、生成与联网工具循环见 app/agent/）

本模块只负责入库与检索；提示词与消息组装在 app/agent/prompts.py。
"""
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.core.embeddings import get_embedder
from app.core.vectorstore import get_vector_store
from app.services.document_loader import load_text
from app.services.embedding_snapshot import get_snapshot
from app.services.splitter import split_text


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


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService()

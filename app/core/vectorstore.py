"""向量库封装：Chroma 本地持久化模式。

向量由 app.core.embeddings 预先算好后传入，Chroma 只负责存储和相似度检索，
数据落盘在 data/chroma/ 目录，重启服务后知识库仍然存在。

两个 collection：
  docs —— 长文档切片（每篇文档多片，metadata 带 doc_id/chunk_index）
  faq  —— 一问一答的 Q&A 对（每条一片，metadata 带 faq_id/question/answer）
两表都用 cosine 相似度，分数可比，因此 query_all 可直接跨表合并排序。
"""
from functools import lru_cache
from typing import Any, Dict, List

from app.config import settings


class VectorStore:
    DOCS_COLLECTION = "docs"
    FAQ_COLLECTION = "faq"

    def __init__(self) -> None:
        self._client = None
        self._collections: Dict[str, Any] = {}

    # ---------- client / collection 基础设施 ----------
    def _ensure_client(self):
        if self._client is None:
            import chromadb

            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        return self._client

    def _get_collection(self, name: str):
        """按名字惰性 get_or_create 一个 cosine collection。"""
        if name not in self._collections:
            client = self._ensure_client()
            self._collections[name] = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    # ---------- docs：长文档切片 ----------
    def add(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        source: str,
    ) -> None:
        """把一篇文档的所有切片写入 docs collection。"""
        col = self._get_collection(self.DOCS_COLLECTION)
        print(f"add doc {doc_id} {len(chunks)} chunks")
        col.add(
            ids=[f"{doc_id}:{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"doc_id": doc_id, "source": source, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )

    def count(self) -> int:
        """返回 docs collection 中的切片总数（启动时用来判断库是否被清空过）。"""
        return self._get_collection(self.DOCS_COLLECTION).count()

    def query(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """按向量相似度检索 docs，返回最相关的若干切片。"""
        col = self._get_collection(self.DOCS_COLLECTION)
        result = col.query(query_embeddings=[embedding], n_results=top_k)
        return self._parse_docs_result(result)

    def delete(self, doc_id: str) -> None:
        """删除某篇文档在 docs collection 中的全部切片。"""
        col = self._get_collection(self.DOCS_COLLECTION)
        col.delete(where={"doc_id": doc_id})

    # ---------- faq：一问一答 ----------
    def add_faq(
        self,
        faq_id: str,
        question: str,
        answer: str,
        source_url: str,
        section: str,
        question_embedding: List[float],
    ) -> None:
        """把一条 FAQ 写入 faq collection。

        注意：只有 question 端的向量（question_embedding），answer 不向量化，
        仅作为 metadata 存储，检索后拼回 text 给 LLM 看。
        """
        col = self._get_collection(self.FAQ_COLLECTION)
        col.add(
            ids=[faq_id],
            # documents 是检索后返回给上游的正文，Q&A 拼一起方便 LLM 引用
            documents=[f"问：{question}\n答：{answer}"],
            embeddings=[question_embedding],
            metadatas=[
                {
                    "faq_id": faq_id,
                    "question": question,
                    "answer": answer,
                    "source_url": source_url,
                    "section": section,
                }
            ],
        )

    def count_faq(self) -> int:
        """返回 faq collection 中的题目总数。"""
        return self._get_collection(self.FAQ_COLLECTION).count()

    def query_faq(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """按向量相似度检索 faq collection。"""
        col = self._get_collection(self.FAQ_COLLECTION)
        result = col.query(query_embeddings=[embedding], n_results=top_k)
        return self._parse_faq_result(result)

    def delete_faq(self, faq_id: str) -> None:
        """删除指定 faq_id 的题目（按 id 精准删，幂等）。"""
        col = self._get_collection(self.FAQ_COLLECTION)
        col.delete(ids=[faq_id])

    # ---------- 综合检索 ----------
    def query_all(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """并行查 docs 和 faq，按 cosine 相似度全局排序后返回 top_k 条。

        每个表各查 top_k * 2，保证合并后有足够候选池；cosine 分数 0~1 跨表可比，
        直接合并降序取前 top_k。每条结果带 collection 字段区分来源。
        """
        # faq 表可能为空（首次启动还没爬），Chroma 对空集合 query 会抛错或返回空，
        # 用 try 兜住，让 docs 检索不受影响
        docs_items: List[Dict[str, Any]] = []
        faq_items: List[Dict[str, Any]] = []
        pool_k = max(top_k * 2, top_k)

        try:
            docs_items = self.query(embedding, pool_k)
        except Exception as e:
            print(f"query docs failed: {e}")

        try:
            faq_items = self.query_faq(embedding, pool_k)
        except Exception as e:
            print(f"query faq failed: {e}")

        merged = docs_items + faq_items
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    # ---------- 内部：解析 Chroma 返回结构 ----------
    @staticmethod
    def _parse_docs_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not result or not result.get("documents") or not result["documents"][0]:
            return items
        for doc, meta, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            items.append(
                {
                    "text": doc,
                    "source": meta.get("source", ""),
                    "doc_id": meta.get("doc_id", ""),
                    # cosine 距离 0~2 → 相似度 0~1
                    "score": round(1 - distance, 4),
                    "collection": "docs",
                }
            )
        return items

    @staticmethod
    def _parse_faq_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not result or not result.get("documents") or not result["documents"][0]:
            return items
        for doc, meta, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            items.append(
                {
                    # documents 字段存的是 "问：…\n答：…"，直接作为正文返回
                    "text": doc,
                    "source": meta.get("source_url", ""),
                    "faq_id": meta.get("faq_id", ""),
                    "question": meta.get("question", ""),
                    "score": round(1 - distance, 4),
                    "collection": "faq",
                }
            )
        return items


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()

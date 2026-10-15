"""稠密检索：包装 VectorStore，复用既有的余弦相似度检索。

只负责"候选生成"这一件事，不做融合、不做重排——那些在 pipeline 里组合。
查询向量由调用方（pipeline）算好传入，避免同一 query 重复向量化。
"""
from typing import List, Optional

from app.core.vectorstore import VectorStore
from app.services.retrieval.base import Candidate


class DenseRetriever:
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def retrieve(
        self,
        query_vector: List[float],
        k: int,
        collections: Optional[tuple] = None,
    ) -> List[Candidate]:
        """按余弦相似度返回 top_k。两表各自召回后按 score 合并——即稠密侧的单一排名。"""
        collections = collections or ("docs", "faq")
        pool = max(k, 1)
        items: List[Candidate] = []
        if "docs" in collections:
            try:
                items += self._store.query(query_vector, pool)
            except Exception:
                pass
        if "faq" in collections:
            try:
                items += self._store.query_faq(query_vector, pool)
            except Exception:
                pass
        items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return items[:pool]

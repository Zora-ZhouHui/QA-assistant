"""检索管线子包：候选生成（稠密/稀疏）→ RRF 融合 → Cross-Encoder 重排。

对外入口：get_retriever()（进程内单例）。
RAGService.search 委托它；agent 层只认"返回一列带 score 的候选"，不感知内部实现。
"""
from app.services.retrieval.pipeline import (
    HybridRetriever,
    build_sparse_index,
    get_retriever,
)

__all__ = ["HybridRetriever", "get_retriever", "build_sparse_index"]

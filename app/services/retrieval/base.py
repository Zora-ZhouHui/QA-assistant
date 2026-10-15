"""检索管线基础契约：候选结构约定 + Retriever / Reranker 协议。

候选统一用 dict 表示（与既有 kb_search 返回结构兼容），键约定：
  id         —— 跨路召回的唯一键：faq 用 faq_id，docs 用 "doc_id:chunk_index"
  text       —— 喂给模型 / 重排器的正文（faq 为"问：…答：…"，docs 为切片正文）
  source     —— 展示用来源（faq 为 source_url，docs 为相对路径，web 为 URL）
  score      —— 对外展示的 0~1 相关度（余弦相似度 / 重排概率）
  collection —— docs | faq | web
  其余字段（faq_id / doc_id / question / title / chunk_index）按需保留。

设计目标：把"候选生成"（稠密 / 稀疏）与"重排"做成可插拔组件，
新增一路召回或换一个重排模型，都不需要改上层（rag / kb_search / agent loop）。
"""
from typing import Any, Dict, List, Optional, Protocol

Candidate = Dict[str, Any]


def candidate_id(item: Candidate) -> str:
    """取候选的唯一键；缺 id 时按 faq_id / doc_id / 正文前缀兜底。"""
    return (
        item.get("id")
        or item.get("faq_id")
        or item.get("doc_id")
        or (item.get("text") or "")[:64]
    )


class Retriever(Protocol):
    """候选生成器：给定 query 返回按相关性降序的候选列表。"""

    def retrieve(
        self, query: str, k: int, collections: Optional[tuple] = None
    ) -> List[Candidate]:
        ...


class Reranker(Protocol):
    """重排器：对候选做精排，返回新的 top_k（分数语义由实现定义，统一归一到 0~1）。"""

    def rerank(
        self, query: str, candidates: List[Candidate], top_k: int
    ) -> List[Candidate]:
        ...

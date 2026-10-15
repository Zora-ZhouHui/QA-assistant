"""稀疏检索：BM25（jieba 中文分词）。

为什么稀疏能补稠密的短板：
  BM25 基于词面匹配，对"专有名词、缩写、精确术语"的召回远强于纯向量，
  与稠密向量互补——这正是混合检索的收益来源。

依赖懒加载：rank_bm25 / jieba 未安装时本模块不报错，
build() 返回 False、hybrid 自动退化为纯稠密（记日志），符合项目的降级哲学。
"""
import logging
from typing import Dict, List, Optional

from app.services.retrieval.base import Candidate, candidate_id

logger = logging.getLogger("app.retrieval")


def _tokenize(text: str) -> List[str]:
    """jieba 分词 + 小写化，作为 BM25 的词项。jieba 不可用时报错由调用方兜住。"""
    import jieba

    return [t.strip() for t in jieba.lcut((text or "").lower()) if t.strip()]


class BM25Index:
    """FAQ 题面 + 文档切片的内存倒排索引（题库量级很小，全内存足够）。"""

    def __init__(self) -> None:
        self._bm25 = None
        self._items: List[Candidate] = []
        self._built = False
        self._last_error: str = ""

    def is_built(self) -> bool:
        return self._built

    @property
    def last_error(self) -> str:
        return self._last_error

    def build(self, items: List[Candidate]) -> bool:
        """用全量条目重建索引；成功返回 True，依赖缺失/空语料返回 False。"""
        self._items = list(items)
        self._built = False
        self._last_error = ""
        try:
            from rank_bm25 import BM25Okapi

            corpus = [_tokenize(it.get("text", "")) for it in self._items]
            self._bm25 = BM25Okapi(corpus) if corpus else None
        except Exception as e:  # 依赖缺失 / 分词失败
            self._bm25 = None
            self._last_error = f"{type(e).__name__}: {e}"
            logger.warning("BM25 索引构建失败，混合检索将退化为纯稠密：%s", self._last_error)
            return False
        self._built = self._bm25 is not None
        return self._built

    def search(
        self, query: str, k: int, collections: Optional[tuple] = None
    ) -> List[Candidate]:
        """按 BM25 降序返回 top_k 条候选；collections 限定可缩小到单表。"""
        if not self._bm25 or not self._items:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Candidate] = []
        for i in order:
            item = self._items[i]
            if collections and item.get("collection") not in collections:
                continue
            cand = dict(item)
            cand["bm25_score"] = float(scores[i])
            out.append(cand)
            if len(out) >= k:
                break
        return out


_index = BM25Index()


def get_bm25_index() -> BM25Index:
    """进程内单例：与 get_vector_store / get_embedder 同构。"""
    return _index

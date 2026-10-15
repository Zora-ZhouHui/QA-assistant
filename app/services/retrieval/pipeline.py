"""检索管线：候选池构建（稠密 + 稀疏并集）→ Cross-Encoder 重排。

对外唯一入口是 retrieve()，上层（rag.RAGService.search）只调它，
kb_search 工具与 agent loop 完全不知道内部有几路召回、是否重排。

为什么不用 RRF 直接改排序（实测结论，勿回退）：
  本项目稠密向量已经很强（评测 L1 95% / L4 84%），而 RRF 会把"双路都命中"
  的候选额外加权。实测出现过一条"BM25 第 18 名 + 稠密第 11 名"的噪声候选
  被叠加分 0.0205 顶到稠密第 1 名（0.0164）之上，L4 反而从 84% 掉到 71%。
  因此这里的定位是：**稀疏一路只负责"扩大候选池（召回）"，不改变排序**；
  最终顺序交给重排器。这也是"retrieve-then-rerank"的标准形态。

降级策略（与项目整体一致）：
  - retrieval_mode=dense            → 只走稠密，退化为改造前的行为；
  - BM25 依赖缺失 / 构建失败         → 只用稠密；
  - 重排模型缺失 / 加载失败          → 记日志并跳过重排，回退到稠密序；
  任何一步失败最差情况都等于旧实现，绝不比改造前更差。
"""
import logging
from functools import lru_cache
from typing import List, Optional

from app.config import settings
from app.core.embeddings import get_embedder
from app.core.vectorstore import get_vector_store
from app.services.retrieval.base import Candidate, candidate_id
from app.services.retrieval.dense import DenseRetriever
from app.services.retrieval.rerank import RerankUnavailable, get_reranker
from app.services.retrieval.sparse import get_bm25_index

logger = logging.getLogger("app.retrieval")


def _dot(a: List[float], b: List[float]) -> float:
    """归一化向量的余弦相似度=点积（embedder 输出已 normalize）。"""
    return sum(x * y for x, y in zip(a, b))


class HybridRetriever:
    def __init__(self) -> None:
        self._store = get_vector_store()
        self._embedder = get_embedder()
        self._dense = DenseRetriever(self._store)
        self._sparse = get_bm25_index()

    # ---------- 稀疏索引 ----------
    def ensure_sparse(self) -> bool:
        """确保 BM25 索引已构建（进程内只建一次）。返回是否可用。"""
        if self._sparse.is_built():
            return True
        items = self._store.iter_all()
        ok = self._sparse.build(items)
        if ok:
            logger.info("BM25 稀疏索引构建完成：%d 条", len(items))
        return ok

    # ---------- 主流程 ----------
    def retrieve(
        self,
        query: str,
        top_k: int,
        collections: Optional[tuple] = None,
        mode: Optional[str] = None,
        use_rerank: Optional[bool] = None,
    ) -> List[Candidate]:
        collections = collections or ("docs", "faq")
        mode = (mode or settings.retrieval_mode or "dense").strip().lower()
        use_rerank = settings.rerank_enabled if use_rerank is None else use_rerank

        # 查询向量只算一次，稠密侧与"补分"复用
        query_vec = self._embedder.embed_query(query)

        pool: List[Candidate] = self._dense.retrieve(
            query_vec, settings.retrieval_candidate_k, collections
        )

        # 稀疏只扩池：把稠密没召回、但词面精确命中的候选补进池子，
        # 让重排器有机会把它们提上来（不改动已有候选的稠密排序）
        if mode == "hybrid" and self.ensure_sparse():
            seen = {candidate_id(it) for it in pool}
            for item in self._sparse.search(
                query, settings.retrieval_candidate_k, collections
            ):
                if candidate_id(item) not in seen:
                    seen.add(candidate_id(item))
                    pool.append(item)

        # 展示用分：稀疏独有候选没有余弦分，按其"与稠密同口径"的文本补算，
        # 保证前端看到的 score 始终是可比的 0~1 相似度
        self._fill_cosine(pool, query_vec)

        # 重排覆盖"全部候选池"（池子本身已被 candidate_k 限制）
        if use_rerank and pool:
            try:
                return get_reranker().rerank(query, pool, top_k)
            except RerankUnavailable as e:
                logger.warning("重排不可用，降级为稠密排序：%s", e)

        # 无重排：按稠密余弦序返回（等价改造前行为，不因加入稀疏而回归）
        pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return pool[:top_k]

    # ---------- 内部 ----------
    def _fill_cosine(self, items: List[Candidate], query_vec: List[float]) -> None:
        """给缺 score 的候选补余弦相似度（批量向量化其检索侧文本）。"""
        missing = [it for it in items if not isinstance(it.get("score"), (int, float))]
        if not missing:
            return
        texts = [it.get("question") or it.get("text", "") for it in missing]
        try:
            vecs = self._embedder.embed_documents(texts)
        except Exception:
            return
        for it, vec in zip(missing, vecs):
            it["score"] = round(_dot(query_vec, vec), 4)


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


def build_sparse_index() -> bool:
    """启动时预热 BM25 索引（在 lifespan 里放线程池调用）。

    只有 hybrid 模式才会用到稀疏扩池；dense 模式跳过构建，避免无谓的
    中文分词与建索引开销（重排关闭时稀疏对 top_k 结果无影响）。
    """
    if (settings.retrieval_mode or "dense").strip().lower() != "hybrid":
        logger.info("检索模式非 hybrid，跳过 BM25 索引预热")
        return False
    return get_retriever().ensure_sparse()

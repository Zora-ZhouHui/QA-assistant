"""评测指标：纯函数，不依赖任何外部服务，方便单测与复用。

先给出检索层的统一约定；生成层 / 引用层的聚合指标见文件后半部分。

统一约定：
  pred_ids —— 按相似度降序返回的候选 id 列表（长度 <= top_k）
  gold_ids / gold_id —— 评测集标注的正确答案 id

本项目每道题只有一个 gold faq_id，因此 Recall@K 与 Hit@K 数值相同；
多正确答案场景下二者才会区分（Recall 看占比，Hit 看是否至少命中一个）。
"""
import math
from typing import Sequence


def recall_at_k(pred_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Recall@K：正确答案被召回的占比。

    单 gold 时退化为 0/1；多 gold 时是 [0, 1] 之间的比例。
    """
    if not gold_ids:
        return 0.0
    return sum(1 for g in gold_ids if g in pred_ids) / len(gold_ids)


def hit_at_k(pred_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Hit@K：top-k 里是否至少命中一个正确答案，返回 0 或 1。"""
    if not gold_ids:
        return 0.0
    return 1.0 if any(g in pred_ids for g in gold_ids) else 0.0


def mrr(pred_ids: Sequence[str], gold_id: str) -> float:
    """MRR：第一个命中的正确答案排名的倒数，未命中为 0。

    排名从 1 开始（第 1 名 → 1.0，第 2 名 → 0.5，…），
    逐 query 求值后由调用方对全量样本取平均。
    """
    for rank, pid in enumerate(pred_ids, start=1):
        if pid == gold_id:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_relevances: Sequence[float], k: int) -> float:
    """nDCG@K：折损累计增益，相关项排名越靠前、得分越高。

    DCG 用 log2(i+1) 折损位置（第 1 名权重 1，第 2 名 1/1.58 …），
    IDCG 是理想排序（所有相关项置顶）下的 DCG，nDCG = DCG / IDCG。
    因此「第一名命中」比「第 k 名命中」给出更接近 1 的分数，对排序好坏有判别力。
    """
    if k <= 0:
        return 0.0
    rels = [float(r) for r in ranked_relevances][:k]
    if not rels:
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0.0 else 0.0


# ---------- 生成层 / 引用层：把裁判返回的计数聚合成分数 ----------

def mean(values: Sequence[float]) -> float:
    """算术平均，空序列返回 0。"""
    return sum(values) / len(values) if values else 0.0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """两向量余弦相似度（归一化向量下退化为点积）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def faithfulness(supported: int, total: int) -> float:
    """Faithfulness = 回答中被资料支撑的原子陈述占比。"""
    return supported / total if total else 0.0


def answer_relevancy(similarities: Sequence[float]) -> float:
    """Answer Relevancy = 由回答反推的问题与原问题相似度的均值。"""
    return mean(similarities)


def citation_precision(supported_cited: int, cited_total: int) -> float:
    """Citation Precision = 带引用的陈述里真的被所引资料支撑的比例。"""
    return supported_cited / cited_total if cited_total else 0.0


def citation_recall(attributable_cited: int, attributable_total: int) -> float:
    """Citation Recall = 需要事实支撑的陈述里真的给了引用的比例。"""
    return attributable_cited / attributable_total if attributable_total else 0.0


def f1(precision: float, recall: float) -> float:
    """精确率与召回率的调和平均。"""
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
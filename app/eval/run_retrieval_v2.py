"""检索层评测 v2：判别力优先，四级难度 + 随机下界基线。

与旧版（run_retrieval.py）的核心区别：
  1) 评测对象不是「题目原文 self-retrieve」（虚高≈100%），而是 build_retrieval_set
     生成的四级难度问法：L1 同义改写 / L2 口语化 / L3 难负例 / L4 激进改写（+ 向量近邻干扰项）。
  2) 指标不止「id 命中」：nDCG@K 对排序敏感，Recall@1 是第 1 名命中率，
     L3 / L4 再用「单选题选对率」让 gold 与近邻干扰项强制正面对决；
     并用「随机检索」作为下界基线，看真实系统相对随机到底有多少判别力。

主对象是 query_faq（纯 FAQ 检索，隔离 docs 干扰）；query_all 的文档挤位归因仍由
run_retrieval.py 的 diff_query_paths 承担，这里不重复。

用法：
    python -m app.eval.build_retrieval_set --n 50 --seed 42   # 一次性生成评测集
    python -m app.eval.run_retrieval_v2 --top-k 4             # 评测
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.core.embeddings import get_embedder
from app.core.vectorstore import get_vector_store
from app.eval.metrics import (cosine_similarity, hit_at_k, mean, mrr,
                              ndcg_at_k, recall_at_k)
from app.services.faq_loader import load_all_faqs


def _ranked_relevances(pred_ids: List[str], gold_id: str) -> List[float]:
    """按检索返回顺序生成 0/1 相关性标注（命中 gold 为 1）。"""
    return [1.0 if pid == gold_id else 0.0 for pid in pred_ids]


def score_sample(sample: Dict[str, Any], top_k: int, store, embedder) -> Dict[str, Any]:
    """对单条样本跑一次 query_faq 检索，返回该样本的各指标与候选序列。"""
    vec = embedder.embed_query(sample["query"])
    items = store.query_faq(vec, top_k)
    pred_ids = [it["faq_id"] for it in items]
    gold = sample["faq_id"]
    rels = _ranked_relevances(pred_ids, gold)
    return {
        "recall": recall_at_k(pred_ids, [gold]),
        "hit": hit_at_k(pred_ids, [gold]),
        "mrr": mrr(pred_ids, gold),
        "rank1": 1.0 if pred_ids and pred_ids[0] == gold else 0.0,
        "ndcg": ndcg_at_k(rels, top_k),
        "pred_ids": pred_ids,
        "gold": gold,
    }


def pick_one_score(
    sample: Dict[str, Any], faq_questions: Dict[str, str], embedder
) -> float:
    """单选题选对率：候选池 = gold + 负例，query 与各候选题目算向量相似度，top1 == gold 记 1。

    这才是真正有判别力的口径：让 gold 和语义极近的干扰项在同一个候选池里硬碰硬，
    而不是在大库里"只要挤进 top_k 就算过"。返回 None 表示无法构成单选（负例不足）。
    """
    gold = sample["faq_id"]
    pool: List[str] = []
    seen = set()
    for fid in [gold] + sample.get("negatives", []):
        if fid in faq_questions and fid not in seen:
            seen.add(fid)
            pool.append(fid)
    if len(pool) < 2:
        return None

    qv = embedder.embed_query(sample["query"])
    vecs = embedder.embed_documents([faq_questions[fid] for fid in pool])
    scores = [cosine_similarity(qv, v) for v in vecs]
    best = pool[max(range(len(scores)), key=lambda i: scores[i])]
    return 1.0 if best == gold else 0.0


def negative_discrimination(sample: Dict[str, Any], pred_ids: List[str]) -> float:
    """难负例判别：gold 是否排在所有负例之前（负例未进 top_k 视为不构成威胁）。

    返回 1.0（gold 压过全部负例）或 0.0（存在负例排在 gold 之前，或 gold 未召回）。
    """
    negs = sample.get("negatives", [])
    if not negs:
        return 1.0
    gold = sample["faq_id"]
    if gold not in pred_ids:
        return 0.0
    gold_rank = pred_ids.index(gold) + 1
    for neg in negs:
        if neg in pred_ids and pred_ids.index(neg) + 1 < gold_rank:
            return 0.0
    return 1.0


def summarize(
    samples: List[Dict[str, Any]], top_k: int, store, faq_questions: Dict[str, str]
) -> Dict[str, Any]:
    """对一批样本逐条打分并聚合，返回各指标均值与诊断明细。"""
    embedder = get_embedder()
    acc: Dict[str, List[float]] = defaultdict(list)
    disc: List[float] = []
    pick: List[float] = []
    gold_ranks: List[int] = []
    surfaced_negs: int = 0
    misses: List[Dict[str, Any]] = []

    for s in samples:
        r = score_sample(s, top_k, store, embedder)
        for key in ("recall", "hit", "mrr", "rank1", "ndcg"):
            acc[key].append(r[key])

        disc.append(negative_discrimination(s, r["pred_ids"]))
        if r["gold"] in r["pred_ids"]:
            gold_ranks.append(r["pred_ids"].index(r["gold"]) + 1)
        else:
            misses.append({"query": s.get("query", ""), "source": s.get("source", "")})

        # 诊断：负例里有几条被错误地拉到 top_k 内
        surfaced_negs += sum(1 for n in s.get("negatives", []) if n in r["pred_ids"])

        # 单选题选对率（仅对带负例的样本）
        pa = pick_one_score(s, faq_questions, embedder)
        if pa is not None:
            pick.append(pa)

    return {
        "n": len(samples),
        **{key: mean(v) for key, v in acc.items()},
        "discrimination": mean(disc),
        "pick_one": mean(pick),
        "gold_rank_mean": mean(gold_ranks),
        "surfaced_negatives": surfaced_negs,
        "miss_count": len(misses),
    }


def random_baseline(n_pool: int, k: int, pick_size: int) -> Dict[str, float]:
    """随机检索的期望分（闭式解），用作判别力下界。

    池子 n_pool 条、1 个 gold，随机打乱时的期望：
      Recall@K / Hit@K = K / N；
      MRR = (1/N) Σ 1/r；
      nDCG@K = (1/N) Σ_{r=1..K} 1/log2(r+1)（单 gold 时 IDCG=1）；
      Recall@1 = 1/N；
      单选题选对率 = 1/pick_size（pick_size = 1 + 负例数）。
    """
    if n_pool <= 0:
        return {}
    recall = hit = min(k / n_pool, 1.0)
    mrr_val = sum(1.0 / r for r in range(1, n_pool + 1)) / n_pool
    scale = min(k, n_pool)
    ndcg = sum(1.0 / math.log2(r + 1) for r in range(1, scale + 1)) / n_pool
    return {
        "recall": recall,
        "hit": hit,
        "mrr": mrr_val,
        "rank1": 1.0 / n_pool,
        "ndcg": ndcg,
        "pick_one": 1.0 / pick_size if pick_size else 0.0,
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="检索层评测 v2（判别力版）")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="报告的最大 K")
    parser.add_argument(
        "--set",
        type=str,
        default=str(settings.data_dir / "eval_retrieval_set.json"),
        help="评测集缓存文件（默认 data/eval_retrieval_set.json）",
    )
    args = parser.parse_args()

    set_path = Path(args.set)
    if not set_path.exists():
        print("评测集不存在：", set_path)
        print("请先运行：python -m app.eval.build_retrieval_set --n 50 --seed 42")
        return

    with open(set_path, encoding="utf-8") as f:
        samples = json.load(f)
    if not samples:
        print("评测集为空。")
        return

    store = get_vector_store()
    n_pool = store.count_faq()
    faq_questions = {r.faq_id: r.question for r in load_all_faqs()}

    by_level: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_level[s.get("level", "?")].append(s)

    # 单选题池大小 = 1(gold) + 负例数（从 L3 样本读众数）
    neg_counts = [len(s.get("negatives", [])) for s in samples if s.get("negatives")]
    n_neg = max(set(neg_counts), key=neg_counts.count) if neg_counts else 5
    pick_size = 1 + n_neg

    print("=" * 60)
    print("检索层评测 v2（判别力版）")
    print("=" * 60)
    print(f"评测集：{len(samples)} 条（每道 gold 题 L1/L2/L3/L4 各 1 条）")
    print(f"FAQ 库总量 N={n_pool}，检索 K={args.top_k}")
    print()

    base = random_baseline(n_pool, args.top_k, pick_size)
    print("【0. 随机下界基线（判断力校准：真实系统应显著高于此）】")
    print(f"    Recall@1         = {_fmt_pct(base['rank1'])}   （=1/{n_pool}）")
    print(f"    Recall@{args.top_k}   = {_fmt_pct(base['recall'])}   （={args.top_k}/{n_pool}）")
    print(f"    nDCG@{args.top_k}    = {_fmt_pct(base['ndcg'])}")
    print(f"    单选选对率（{pick_size}选1）= {_fmt_pct(base['pick_one'])}   （=1/{pick_size}）")
    print()

    level_results: Dict[str, Dict[str, Any]] = {}
    for level, title in [("L1", "同义改写"), ("L2", "口语化"), ("L3", "难负例"), ("L4", "激进改写")]:
        if level not in by_level:
            continue
        res = summarize(by_level[level], args.top_k, store, faq_questions)
        level_results[level] = res
        print(f"【{level} · {title}】")
        print(f"    Recall@1       = {_fmt_pct(res['rank1'])}")
        print(f"    Recall@{args.top_k}   = {_fmt_pct(res['recall'])}")
        print(f"    MRR            = {_fmt_pct(res['mrr'])}")
        print(f"    nDCG@{args.top_k}    = {_fmt_pct(res['ndcg'])}")
        if level in ("L3", "L4"):
            print(f"    单选选对率     = {_fmt_pct(res['pick_one'])}"
                  f"（{pick_size} 选 1，随机基线 {_fmt_pct(base['pick_one'])}）")
            print(f"    负例判别率     = {_fmt_pct(res['discrimination'])}"
                  f"（gold 压过全部近邻干扰项的比例）")
            print(f"    gold 平均排名  = {res['gold_rank_mean']:.2f}（越接近 1 越好）")
            print(f"    被误拉进 top_k 的负例数 = {res['surfaced_negatives']}")
        print(f"    未命中（gold 未入 top_k）：{res['miss_count']} 条")
        print()

    print("【判别力小结】")
    print(f"  随机基线 Recall@1 = {_fmt_pct(base['rank1'])}；"
          f"单选基线 = {_fmt_pct(base['pick_one'])}——真实系统越远离该值，评测越有区分度。")
    if "L1" in level_results and "L4" in level_results:
        l1 = level_results["L1"]["rank1"]
        l4 = level_results["L4"]["rank1"]
        print(f"  L1 软改写 Recall@1 {_fmt_pct(l1)} → L4 激进改写 Recall@1 {_fmt_pct(l4)}"
              f"（落差 {_fmt_pct(l1 - l4)}，落差越大评测越能拉开难度）")
    if "L4" in level_results:
        lift = level_results["L4"]["pick_one"] / base["pick_one"] if base["pick_one"] else 0.0
        print(f"  L4 单选选对率 {_fmt_pct(level_results['L4']['pick_one'])}"
              f"相对随机基线提升：{lift:.1f}x")


if __name__ == "__main__":
    main()
"""检索层离线评测入口：零评分成本，输出 Recall@K / Hit@K / MRR 报告。

两个层次，回答不同问题：
  1) 入库正确性 sanity check —— query 用题目原文，验证「每道题都能自检索到」，
     即 faq_id 映射 / embedding 前缀 / 索引是否正确，不衡量检索质量；
  2) 真实检索质量 —— query 用改写后的同义问句（先跑 build_queries 生成），
     衡量 embedding 能否「透过措辞变化认出同一道题」，这才是简历上可写的 Recall@K。

两套检索路径同时评测：
  query_faq  主指标——FAQ 库自身检索质量（隔离 docs 表干扰）
  query_all  参考指标——生产实际路径（docs + faq 混合排序），反映上线后真实召回

用法：
    python -m app.eval.build_queries --n 50          # 一次性：生成改写 query（需 LLM_API_KEY）
    python -m app.eval.run_retrieval                 # 评测（默认 50 题 / top_k=4 / seed=42）
    python -m app.eval.run_retrieval --n 100 --top-k 10
"""
import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List

from app.config import settings
from app.core.embeddings import get_embedder
from app.core.vectorstore import get_vector_store
from app.eval.dataset import sample_faqs
from app.eval.metrics import hit_at_k, mrr, recall_at_k


def extract_ids(items: List[Dict[str, Any]], mode: str) -> List[str]:
    """把一条检索结果转成候选 id 序列（保持排名顺序）。

    query_faq 里每项都有 faq_id；query_all 里 docs 项没有 faq_id，
    用 doc:{doc_id} 占位，这样命中判定时仍能正确识别 faq 项的名次（含 docs 穿插）。
    """
    if mode == "query_faq":
        return [it["faq_id"] for it in items]
    return [it.get("faq_id") or f"doc:{it.get('doc_id')}" for it in items]


def evaluate_queries(samples, mode: str, top_k: int, store, embedder) -> Dict[str, Any]:
    """对一批 (faq_id, query) 样本跑一次检索评测，返回聚合结果。

    samples：List[dict]，每项含 faq_id / query / source。
    """
    hits = [0] * top_k          # 每个 k 的命中题数（供 Recall@K 曲线）
    mrr_sum = 0.0
    per_source: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hit": 0, "total": 0})
    misses: List[tuple] = []

    for s in samples:
        vec = embedder.embed_query(s["query"])
        items = store.query_faq(vec, top_k) if mode == "query_faq" else store.query_all(vec, top_k)
        ranked = extract_ids(items, mode)
        gold = s["faq_id"]

        # 逐 k 累计 Recall@K（单 gold 下等价 Hit@K）
        for k in range(1, top_k + 1):
            if recall_at_k(ranked[:k], [gold]):
                hits[k - 1] += 1
        mrr_sum += mrr(ranked, gold)

        src = s.get("source", "")
        per_source[src]["total"] += 1
        if hit_at_k(ranked, [gold]):
            per_source[src]["hit"] += 1
        else:
            top1_text = items[0].get("text", "")[:40] if items else ""
            misses.append((s, top1_text))

    return {
        "hits": hits,
        "mrr": mrr_sum / len(samples) if samples else 0.0,
        "per_source": {src: dict(v) for src, v in per_source.items()},
        "misses": misses,
    }


def _print_recall_curve(total: int, hits: List[int], label: str) -> None:
    for k in range(1, len(hits) + 1):
        print(f"  {label}{k:<3} = {hits[k - 1] / total * 100:5.2f}%")


def _print_block(title: str, res: Dict[str, Any], total: int) -> None:
    print(title)
    _print_recall_curve(total, res["hits"], "Recall@")
    print(f"  MRR      = {res['mrr'] * 100:5.2f}%")
    print()


def _print_misses(misses: List[tuple]) -> None:
    if not misses:
        print("全部命中，无未命中样例。")
        return
    print("  未命中样例（最多显示 5 条）：")
    for s, top1 in misses[:5]:
        q = (s.get("query") or "")[:30].replace("\n", " ")
        print(f"    [{s.get('source', '')}] 「{q}…」 -> 误召回 top1：{top1}")


def diff_query_paths(samples, top_k, store, embedder) -> Dict[str, Any]:
    """对比同批样本在 query_faq 与 query_all 下的召回，定位文档挤位影响。

    找出「query_faq 命中、query_all 未命中」的样本——即被 docs 完全挤出 top_k 的题目，
    并记录 query_all 里占位的条目（来源 / 分数 / 摘要），把总量差值拆成可归因的明细。
    """
    displaced: List[dict] = []                          # 被完全挤出 top_k 的样本
    demoted: List[dict] = []                            # 仍命中但名次被压低
    displace_source: Dict[str, int] = defaultdict(int)  # 挤位者来源分布

    for s in samples:
        vec = embedder.embed_query(s["query"])
        gold = s["faq_id"]

        faq_items = store.query_faq(vec, top_k)
        all_items = store.query_all(vec, top_k)
        faq_ranked = extract_ids(faq_items, "query_faq")
        all_ranked = extract_ids(all_items, "query_all")

        faq_hit = hit_at_k(faq_ranked, [gold])
        all_hit = hit_at_k(all_ranked, [gold])

        if faq_hit and not all_hit:
            # gold 已被挤出 top_k，query_all 的前 top_k 条都是「排在 gold 之前」的占位者
            blockers = []
            for it in all_items:
                cid = it.get("faq_id") or f"doc:{it.get('doc_id')}"
                blockers.append(
                    {
                        "id": cid,
                        "collection": it.get("collection", ""),
                        "source": it.get("source", ""),
                        "score": it.get("score"),
                        "text": (it.get("text", "") or "")[:40],
                    }
                )
                key = (
                    it.get("source", "")
                    if it.get("collection") == "docs"
                    else f"faq:{it.get('source', '')}"
                )
                displace_source[key] += 1
            displaced.append(
                {
                    "faq_id": gold,
                    "query": s.get("query", ""),
                    "source": s.get("source", ""),
                    "blockers": blockers,
                }
            )
        elif faq_hit and all_hit:
            faq_rank = faq_ranked.index(gold) + 1
            all_rank = all_ranked.index(gold) + 1
            if all_rank > faq_rank:
                demoted.append(
                    {
                        "faq_id": gold,
                        "query": s.get("query", ""),
                        "faq_rank": faq_rank,
                        "all_rank": all_rank,
                    }
                )

    return {"displaced": displaced, "demoted": demoted, "displace_source": dict(displace_source)}


def _print_displacement(diff: Dict[str, Any]) -> None:
    displaced = diff["displaced"]
    demoted = diff["demoted"]
    print("【3. 文档干扰归因（query_faq 命中 → query_all 未命中）】")
    print(f"  被 docs 挤出 top_k 的样本：{len(displaced)} 道")
    print(f"  仍命中但名次被压低：{len(demoted)} 道")
    if not displaced:
        print("  无文档抢位损失，query_all 与 query_faq 无召回落差。")
        return
    print()
    print("  被挤掉样本明细（最多 5 条，占位条目按 query_all 排名）：")
    for d in displaced[:5]:
        q = (d["query"] or "")[:30].replace("\n", " ")
        print(f"    [{d['source']}] 「{q}…」 gold={d['faq_id']}")
        for b in d["blockers"]:
            sid = (b["id"] or "")[:26]
            print(f"        ← {b['collection']:<4} {sid:<26} score={b['score']} 「{b['text']}…」")
    print()
    print("  挤位者来源分布：")
    for src, cnt in sorted(diff["displace_source"].items(), key=lambda kv: -kv[1]):
        print(f"    {src:<28} {cnt} 次")


def _load_paraphrases(records) -> List[Dict[str, str]]:
    """读取 build_queries 生成的改写 query，按 faq_id 对齐到抽样题。"""
    path = settings.data_dir / "eval_queries.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    mapping = {d["faq_id"]: d.get("paraphrase", "") for d in raw}
    samples = []
    for rec in records:
        para = mapping.get(rec.faq_id)
        if para:
            samples.append({"faq_id": rec.faq_id, "query": para, "source": rec.source})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="检索层离线评测（Recall@K / Hit@K / MRR）")
    parser.add_argument("--n", type=int, default=50, help="抽样题数（默认 50）")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="报告的最大 K（默认取 settings.top_k）")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子（默认 42）")
    args = parser.parse_args()

    records = sample_faqs(n=args.n, seed=args.seed)
    if not records:
        print("无法加载 FAQ 数据，请确认 data/faqs.json 是否存在")
        return

    store = get_vector_store()
    embedder = get_embedder()

    source_dist: Dict[str, int] = defaultdict(int)
    for r in records:
        source_dist[r.source] += 1

    print("=" * 60)
    print("检索层离线评测")
    print("=" * 60)
    print(f"数据集：{len(records)} 道（seed={args.seed}），来源分布：{dict(source_dist)}")
    print(f"FAQ 库总量：{store.count_faq()}，检索 K={args.top_k}")
    print()

    # ---- 1. 入库正确性 sanity check（query=原题）----
    sanity_samples = [
        {"faq_id": r.faq_id, "query": r.question, "source": r.source} for r in records
    ]
    sanity = evaluate_queries(sanity_samples, "query_faq", args.top_k, store, embedder)
    print("【1. 入库正确性 sanity check（query=题目原文，仅验证可自检索）】")
    print(f"  Recall@{args.top_k} = {sanity['hits'][args.top_k - 1] / len(records) * 100:5.2f}%  "
          f"MRR = {sanity['mrr'] * 100:5.2f}%")
    print("  （100% 只是说明索引没坏、题目都入库了，不代表检索质量）")
    print()

    # ---- 2. 真实检索质量（query=改写题）----
    para_samples = _load_paraphrases(records)
    if not para_samples:
        print("【2. 真实检索质量（改写 query）】")
        print("  尚未生成改写 query。请先运行：")
        print("    python -m app.eval.build_queries --n %d --seed %d" % (args.n, args.seed))
        print("  （需要环境变量 LLM_API_KEY）")
        return

    aligned = len(para_samples)
    if aligned < len(records):
        print(f"  提示：改写缓存只覆盖 {aligned}/{len(records)} 道抽样题，"
              f"请重跑 build_queries 使 --n/--seed 保持一致。\n")

    print("【2. 真实检索质量（query=改写题，衡量鲁棒性 Recall@K）】")
    for mode, label in [
        ("query_faq", "主指标 · query_faq（纯 FAQ 检索）"),
        ("query_all", "参考指标 · query_all（生产路径 docs+faq 混合）"),
    ]:
        res = evaluate_queries(para_samples, mode, args.top_k, store, embedder)
        _print_block(f"  2a. {label}", res, aligned)

    main_res = evaluate_queries(para_samples, "query_faq", args.top_k, store, embedder)
    print(f"【分源 Recall@{args.top_k}（改写 query → query_faq）】")
    for src, v in sorted(main_res["per_source"].items()):
        pct = v["hit"] / v["total"] * 100 if v["total"] else 0.0
        print(f"  {src:<20} {v['hit']}/{v['total']}  {pct:5.2f}%")
    print()

    _print_misses(main_res["misses"])

    # ---- 3. 文档干扰归因（per-query diff）----
    diff = diff_query_paths(para_samples, args.top_k, store, embedder)
    _print_displacement(diff)


if __name__ == "__main__":
    main()
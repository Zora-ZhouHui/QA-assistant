"""引用层离线评测：Citation Precision / Recall（LLM-as-a-Judge）。

依赖生成层产出——先跑 run_generation，它会生成带 [资料N] 引用的回答并缓存到
data/eval_generation.json；本脚本读缓存，对每份「回答 + 资料」做逐句引用判定：
  Citation Precision  带引用的陈述里，有多少真的被所引资料支撑
  Citation Recall     需要事实支撑的陈述里，有多少真的给了引用

用法（需先有 data/eval_generation.json 且 LLM_API_KEY）：
    python -m app.eval.run_citation
"""
import argparse
import asyncio
import json
from typing import Any, Dict, List

from app.config import settings
from app.eval.judge import judge_citation
from app.eval.metrics import citation_precision, citation_recall, f1, mean


def precision_recall(statements: List[Dict[str, Any]]):
    """从逐句判定里算出单份样本的 Citation Precision / Recall。"""
    cited_supported = 0
    cited_total = 0
    attributable_cited = 0
    attributable_total = 0
    for s in statements:
        if not isinstance(s, dict):
            continue
        cited = s.get("cited") or []
        has_citation = len(cited) > 0
        attributable = bool(s.get("attributable"))
        supported = bool(s.get("supported"))
        if has_citation:
            cited_total += 1
            if supported:
                cited_supported += 1
        if attributable:
            attributable_total += 1
            if has_citation:
                attributable_cited += 1
    p = citation_precision(cited_supported, cited_total)
    r = citation_recall(attributable_cited, attributable_total)
    return p, r


async def run() -> None:
    path = settings.data_dir / "eval_generation.json"
    if not path.exists():
        print("未找到 data/eval_generation.json，请先运行 run_generation")
        return
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    print(f"引用层评测：{len(rows)} 份回答，逐句判定引用质量…")
    precisions: List[float] = []
    recalls: List[float] = []
    failed = 0
    for row in rows:
        try:
            statements = await judge_citation(row["answer"], row["context"])
            p, r = precision_recall(statements)
            precisions.append(p)
            recalls.append(r)
            print(f"  [ok ] {row['faq_id']:<24} precision={p:.2f}  recall={r:.2f}")
        except Exception as e:
            failed += 1
            print(f"  [skip] {row['faq_id']:<24} {e}")

    if not precisions:
        print("\n没有成功样本，无法聚合")
        return
    p = mean(precisions)
    r = mean(recalls)
    print("\n" + "=" * 60)
    print("引用层离线评测结果")
    print("=" * 60)
    print(f"样本：{len(precisions)} 成功 / {failed} 失败（跳过）")
    print(f"Citation Precision = {p * 100:.2f}%")
    print(f"Citation Recall    = {r * 100:.2f}%")
    print(f"Citation F1        = {f1(p, r) * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="引用层评测（Citation Precision / Recall）")
    parser.parse_args()
    asyncio.run(run())


if __name__ == "__main__":
    main()
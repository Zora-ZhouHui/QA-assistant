"""评测数据集构造：从 data/faqs.json 分层抽样出检索层离线评测样本。

检索层评测不依赖 LLM，样本只需「问题 + gold faq_id」即可判定命中；
FaqRecord 里 answer / source_url 等字段保留，未命中时方便人工抽查原文。
"""
import random
from pathlib import Path
from typing import Dict, List, Optional

from app.services.faq_loader import FaqRecord, load_all_faqs


def sample_faqs(
    n: int = 50, seed: int = 42, path: Optional[Path] = None
) -> List[FaqRecord]:
    """按数据源分层抽样 n 道题，保证三个源都有覆盖，seed 固定使结果可复现。

    用「最大余数法」按各源占比分配配额，避免简单 round 导致多抽 / 漏抽；
    某源题量少于配额时会被截断到其总量（小源不会被抽空）。
    """
    records = load_all_faqs(path)
    if not records or n >= len(records):
        return records

    by_source: Dict[str, List[FaqRecord]] = {}
    for rec in records:
        by_source.setdefault(rec.source, []).append(rec)

    # 最大余数法分配配额
    total = len(records)
    quotas: Dict[str, int] = {}
    remainders: List[tuple] = []
    allocated = 0
    for source, items in by_source.items():
        exact = n * len(items) / total
        base = int(exact)
        quotas[source] = min(base, len(items))
        allocated += quotas[source]
        remainders.append((exact - base, source))

    # 余数从大到小补齐差额（被截断的源不会再被补）
    for _, source in sorted(remainders, reverse=True):
        if allocated >= n:
            break
        if quotas[source] < len(by_source[source]):
            quotas[source] += 1
            allocated += 1

    rng = random.Random(seed)
    sampled: List[FaqRecord] = []
    for source, items in by_source.items():
        sampled.extend(rng.sample(items, quotas[source]))
    rng.shuffle(sampled)
    return sampled
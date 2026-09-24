"""生成检索层评测的改写 query（paraphrase），离线一次性，缓存到 data/eval_queries.json。

为什么需要改写：直接用题目原文检索是「自己搜自己」，相似度天然≈1，Recall@1 必然 100%，
没有区分度（BEIR 范式要求 query ≠ 库中段落）。把题目改写成措辞不同的同义问句再去检索，
才能真正检验 embedding 能否「透过措辞变化认出同一道题」，这是业界公认的鲁棒性 Recall@K 测法。

用法：
    python -m app.eval.build_queries --n 50 --seed 42
需要先配好 LLM_API_KEY（走环境变量或 .env）。

产出 JSON 数组，每条 {faq_id, source, question, paraphrase}，
供生成层 / 引用层复用为改写 query，生成后即与运行时解耦（重跑评测无需再调 LLM）。
"""
import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict

from app.config import settings
from app.core.llm import get_llm_client
from app.eval.dataset import sample_faqs

_SYSTEM_PROMPT = (
    "你是面试题库的改写助手。把用户给出的面试题改写成一个语义完全相同、但措辞不同的版本。"
    "要求：1) 保持是同一道题、同一考点，不得增删任何考点；"
    "2) 用词与原文有明显差异（换同义词、换句式、调整语序）；"
    "3) 只输出改写后的问题本身，不要任何解释、引号或多余文字。"
)


async def _paraphrase_one(client, question: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        out = ""
        async for token in client.stream_chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        ):
            out += token
        return out.strip()


async def _paraphrase_all(questions: List[str], concurrency: int) -> List[str]:
    client = get_llm_client()
    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*[_paraphrase_one(client, q, sem) for q in questions])


def main() -> None:
    parser = argparse.ArgumentParser(description="生成检索层评测的改写 query")
    parser.add_argument("--n", type=int, default=50, help="改写题数（默认 50，与评测一致）")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子（默认 42）")
    parser.add_argument("--concurrency", type=int, default=8, help="并发改写数（默认 8）")
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.data_dir / "eval_queries.json",
        help="输出缓存文件（默认 data/eval_queries.json）",
    )
    args = parser.parse_args()

    if not settings.llm_api_key:
        print("未配置 LLM_API_KEY：请先在环境变量或 .env 里填入密钥")
        return

    records = sample_faqs(n=args.n, seed=args.seed)
    if not records:
        print("无法加载 FAQ 数据，请确认 data/faqs.json 是否存在")
        return

    print(f"对 {len(records)} 道题生成改写 query（concurrency={args.concurrency}）…")
    paraphrases = asyncio.run(
        _paraphrase_all([r.question for r in records], args.concurrency)
    )

    data: List[Dict[str, str]] = [
        {
            "faq_id": rec.faq_id,
            "source": rec.source,
            "question": rec.question,
            "paraphrase": p,
        }
        for rec, p in zip(records, paraphrases)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 打印几条样例，便于人工核对改写质量
    print(f"已写入 {args.out}（{len(data)} 条），样例：")
    for d in data[:3]:
        print(f"  原题：{d['question'][:40]}…")
        print(f"  改写：{d['paraphrase'][:40]}…")


if __name__ == "__main__":
    main()
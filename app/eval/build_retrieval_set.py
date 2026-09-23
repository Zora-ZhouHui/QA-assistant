"""构造检索层评测集：三级难度 + 自动负例，离线缓存到 data/eval_retrieval_set.json。

背景：旧评测「直接用题目原文去检索」是"自己搜自己"，相似度天然≈1，Recall@1 必然 100%，
分数虚高、没有判别力；且指标只看"id 命中"，对排序好坏、无关项混入都不敏感。
评测目标应是「透过措辞变化 / 口语化 / 语义相近的干扰项，能否把正确那道题排到最前」。

四级难度（同一道 gold 题，四种检索问法）：
  L1 同义改写 —— LLM 把原题改写成措辞不同、考点相同的问句，检验 embedding 能否"透过措辞认出同一题"
  L2 口语化   —— LLM 改写成真实用户提问的口语（含省略 / 非正式表达），贴近线上分布；
                 定位是「线上分布鲁棒性」档（非难度递增，实测召回不低于 L1）
  L3 难负例   —— 复用 L1 的改写问句，再挖掘"语义相近但考点不同"的向量近邻作负例，
                 检验 embedding 能否把 gold 排在所有干扰项之前
  L4 激进改写 —— 措辞 / 句式 / 语序 / 信息组织尽量拉开、考点不变，逼 embedding 漂移，
                 对比"软改写"与"硬改写"下的检索差距（判别力的直接来源）

负例纯自动（零 LLM）：对改写问句做 query embedding，在全部 FAQ 题面的向量里找最相近的
top_n 个非自身题目，它们就是"最像但不应命中"的干扰项。评测时据此算「gold 是否压过全部负例」。

用法：
    python -m app.eval.build_retrieval_set --n 50 --seed 42
L1/L2 改写需要 LLM_API_KEY；负例挖掘需要本地 embedding 模型。
产出 JSON 数组，每条 {faq_id, source, question, level, query, gold_ids, negatives}。
"""
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple

from app.config import settings
from app.core.embeddings import get_embedder
from app.core.llm import get_llm_client
from app.eval.dataset import sample_faqs
from app.eval.metrics import cosine_similarity
from app.services.faq_loader import FaqRecord, load_all_faqs

_PARAPHRASE_PROMPT = (
    "你是面试题库的改写助手。把用户给出的面试题改写成一个语义完全相同、但措辞不同的版本。"
    "要求：1) 保持是同一道题、同一考点，不得增删任何考点；"
    "2) 用词与原文有明显差异（换同义词、换句式、调整语序）；"
    "3) 只输出改写后的问题本身，不要任何解释、引号或多余文字。"
)

_COLLOQUIAL_PROMPT = (
    "你是面试题库的口语化助手。把用户给出的面试题改写成求职者真实的问法："
    "用口语、可带省略和口语化语气词，保持考点不变，但像真人随手打出的提问。"
    "要求：1) 考点完全相同，不得增删；2) 语气明显口语化（区别于书面改写）；"
    "3) 只输出口语化问句本身，不要任何解释、引号或多余文字。"
)

_HARD_PARAPHRASE_PROMPT = (
    "你是面试题库的改写助手。把用户给出的面试题改写成一个「考察完全相同的知识点、"
    "正确答案完全不变，但措辞与表述方式尽可能不同」的版本。"
    "要求：1) 考察的知识点和正确答案必须 100% 不变，只能换一种问法；"
    "2) 尽量与原题拉开距离：换同义词、改句式、调整语序、重组信息、换提问角度，"
    "避免照搬原文任何短语；"
    "3) 可更口语也可更书面，只要仍是同一考点即可；"
    "4) 只输出改写后的问题本身，不要任何解释、引号或多余文字。"
)


async def _rewrite_one(
    client, question: str, system_prompt: str, sem: asyncio.Semaphore
) -> str:
    async with sem:
        out = ""
        async for token in client.stream_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        ):
            out += token
        return out.strip()


async def _rewrite_all(
    client, questions: List[str], system_prompt: str, concurrency: int
) -> List[str]:
    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(
        *[_rewrite_one(client, q, system_prompt, sem) for q in questions]
    )


def _build_neighbor_index(records: List[FaqRecord]) -> Tuple[List[str], List[List[float]]]:
    """对全量 FAQ 题面批量 embedding，返回 (id 列表, 对应向量)。

    题面用 embed_documents（无检索指令），与生产入库到 Chroma 时的方式一致，
    保证难负例的相似度口径和真实检索一致。
    """
    ids = [r.faq_id for r in records]
    questions = [r.question for r in records]
    vecs = get_embedder().embed_documents(questions)
    return ids, vecs


def _top_negatives(
    query: str,
    ids: List[str],
    vecs: List[List[float]],
    gold_id: str,
    top_n: int,
) -> List[str]:
    """返回与改写问句最相近的 top_n 个非 gold 题目 id（难负例）。

    用 embed_query（带检索指令）对问句向量化，与全量题面算余弦相似度，
    取除 gold 外最像的若干条 —— 它们就是"最像但不应命中"的干扰项。
    """
    if top_n <= 0:
        return []
    qv = get_embedder().embed_query(query)
    scored = [
        (faq_id, cosine_similarity(qv, vec))
        for faq_id, vec in zip(ids, vecs)
        if faq_id != gold_id
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [faq_id for faq_id, _ in scored[:top_n]]


def main() -> None:
    parser = argparse.ArgumentParser(description="构造检索层评测集（三级难度 + 负例）")
    parser.add_argument("--n", type=int, default=50, help="抽样题数（默认 50）")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子（默认 42）")
    parser.add_argument("--concurrency", type=int, default=8, help="并发改写数（默认 8）")
    parser.add_argument("--negatives", type=int, default=5, help="每题难负例数（默认 5）")
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.data_dir / "eval_retrieval_set.json",
        help="输出缓存文件（默认 data/eval_retrieval_set.json）",
    )
    args = parser.parse_args()

    if not settings.llm_api_key:
        print("未配置 LLM_API_KEY：请先在环境变量或 .env 里填入密钥")
        return

    records = sample_faqs(n=args.n, seed=args.seed)
    if not records:
        print("无法加载 FAQ 数据，请确认 data/faqs.json 是否存在")
        return

    client = get_llm_client()
    questions = [r.question for r in records]
    print(
        f"对 {len(questions)} 道题生成 L1 同义 / L2 口语化 / L4 激进改写"
        f"（concurrency={args.concurrency}）…"
    )
    paraphrases = asyncio.run(
        _rewrite_all(client, questions, _PARAPHRASE_PROMPT, args.concurrency)
    )
    colloquials = asyncio.run(
        _rewrite_all(client, questions, _COLLOQUIAL_PROMPT, args.concurrency)
    )
    hard_paraphrases = asyncio.run(
        _rewrite_all(client, questions, _HARD_PARAPHRASE_PROMPT, args.concurrency)
    )

    # 负例挖掘基于全量 FAQ，而非抽样子集（干扰项应从整个题库里找）
    all_records = load_all_faqs()
    print(f"对全量 {len(all_records)} 道题做 embedding，用于挖掘难负例…")
    ids, vecs = _build_neighbor_index(all_records)

    samples: List[Dict] = []
    for rec, para, colloq, hard in zip(records, paraphrases, colloquials, hard_paraphrases):
        base = {
            "faq_id": rec.faq_id,
            "source": rec.source,
            "question": rec.question,
            "gold_ids": [rec.faq_id],
        }
        samples.append({**base, "level": "L1", "query": para, "negatives": []})
        samples.append({**base, "level": "L2", "query": colloq, "negatives": []})
        negs = _top_negatives(para, ids, vecs, rec.faq_id, args.negatives)
        samples.append({**base, "level": "L3", "query": para, "negatives": negs})
        hard_negs = _top_negatives(hard, ids, vecs, rec.faq_id, args.negatives)
        samples.append({**base, "level": "L4", "query": hard, "negatives": hard_negs})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 打印样例便于人工核对改写质量与负例合理性
    print(f"已写入 {args.out}（{len(samples)} 条，每题 L1/L2/L3/L4 各 1 条），样例：")
    for s in samples[:3]:
        print(f"  [{s['level']}] 原题：{s['question'][:30]}…")
        print(f"        query：{s['query'][:40]}…  负例 {len(s['negatives'])} 个")


if __name__ == "__main__":
    main()
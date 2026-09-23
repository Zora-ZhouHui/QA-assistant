"""生成层离线评测：Faithfulness / Answer Relevancy（LLM-as-a-Judge）。

流程（每道抽样题）：
  1. 取改写 query（build_queries 生成，缺失则退回原题）做真实检索；
  2. query_all 取 top_k 资料，编号成 [资料1..k]；
  3. 让模型「只依据资料」生成带 [资料N] 引用的回答；
  4. 裁判打分：
       Faithfulness    —— 回答拆成原子陈述，逐条判断是否被资料支撑（检测幻觉）
       Answer Relevancy —— 由回答反推 3 个问题，与原问题做 bge 向量相似度取均值
  5. 结果缓存到 data/eval_generation.json，供引用层（run_citation）复用。

为什么走 query_all：与生产一致（docs + faq 混合）。FAQ 的 faq 条目文本天然带答案，
因此本层 Faithfulness 的基线会偏高——它衡量的是「模型有没有在此基础上再多编内容」，
仍能暴露幻觉类回归，只是"答错且未引用"这类错误不出现在这里。

用法（需 LLM_API_KEY）：
    python -m app.eval.run_generation --n 20
"""
import argparse
import asyncio
import json
from typing import Any, Dict, List

from app.config import settings
from app.core.embeddings import get_embedder
from app.core.vectorstore import get_vector_store
from app.eval.dataset import sample_faqs
from app.eval.judge import call_llm, judge_faithfulness, judge_relevancy_questions
from app.eval.metrics import answer_relevancy, cosine_similarity, faithfulness, mean

_GENERATION_PROMPT = """你是知识问答助手。请只根据下面提供的「资料」回答问题，不要使用资料之外的知识。回答要简洁准确；凡是用到某条资料支撑结论的句子，句末标注来源编号，格式 [资料N]，多个用逗号分隔。引用编号必须真实存在；资料不足以回答时如实说明，不要编造。

<资料>
{context}

<问题>
{question}"""


def format_context(items: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[资料{i + 1}] {it['text']}" for i, it in enumerate(items))


def compute_relevancy(gen_questions: List[str], original_q: str, embedder) -> float:
    """Answer Relevancy = 由回答反推的问题与原问题的向量相似度均值（RAGAS 式）。"""
    if not gen_questions:
        return 0.0
    orig_vec = embedder.embed_query(original_q)
    sims = [cosine_similarity(orig_vec, embedder.embed_query(q)) for q in gen_questions]
    return answer_relevancy(sims)


async def process_one(rec, query_text: str, top_k: int, store, embedder) -> Dict[str, Any]:
    vec = embedder.embed_query(query_text)
    items = store.query_all(vec, top_k)
    context_text = format_context(items)

    answer = await call_llm([
        {"role": "user", "content": _GENERATION_PROMPT.format(context=context_text, question=query_text)},
    ])

    faith = await judge_faithfulness(answer, context_text)
    rel_qs = await judge_relevancy_questions(answer)
    rel = compute_relevancy(rel_qs, query_text, embedder)

    return {
        "faq_id": rec.faq_id,
        "source": rec.source,
        "question": rec.question,
        "query": query_text,
        "answer": answer,
        "context": context_text,
        "faithfulness": faithfulness(faith["supported"], faith["total"]),
        "faithfulness_supported": faith["supported"],
        "faithfulness_total": faith["total"],
        "answer_relevancy": rel,
    }


def _load_paraphrases() -> Dict[str, str]:
    path = settings.data_dir / "eval_queries.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {d["faq_id"]: d.get("paraphrase") or d["question"] for d in raw}


async def run(n: int) -> None:
    records = sample_faqs(n=n, seed=42)
    if not records:
        print("无法加载 FAQ 数据，请确认 data/faqs.json 是否存在")
        return
    paras = _load_paraphrases()
    store = get_vector_store()
    embedder = get_embedder()

    print(f"生成层评测：{len(records)} 道题，生成回答并裁判打分…")
    rows: List[Dict[str, Any]] = []
    failed = 0
    for rec in records:
        query_text = paras.get(rec.faq_id, rec.question)
        try:
            row = await process_one(rec, query_text, settings.top_k, store, embedder)
            rows.append(row)
            print(
                f"  [ok ] {rec.faq_id:<24} faithfulness={row['faithfulness']:.2f}  relevancy={row['answer_relevancy']:.2f}"
            )
        except Exception as e:
            failed += 1
            print(f"  [skip] {rec.faq_id:<24} {e}")

    out = settings.data_dir / "eval_generation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    f_vals = [r["faithfulness"] for r in rows if r["faithfulness_total"] > 0]
    r_vals = [r["answer_relevancy"] for r in rows]
    print("\n" + "=" * 60)
    print("生成层离线评测结果")
    print("=" * 60)
    print(f"样本：{len(rows)} 成功 / {failed} 失败（跳过）")
    print(f"Faithfulness     = {mean(f_vals) * 100:.2f}%（{len(f_vals)} 个有效样本）")
    print(f"Answer Relevancy = {mean(r_vals) * 100:.2f}%")
    print(f"\n结果已缓存到 {out}，供 run_citation 复用")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成层评测（Faithfulness / Answer Relevancy）")
    parser.add_argument("--n", type=int, default=20, help="抽样题数（默认 20，每道需多次 LLM 调用）")
    args = parser.parse_args()
    asyncio.run(run(args.n))


if __name__ == "__main__":
    main()
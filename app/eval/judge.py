"""LLM-as-a-Judge 裁判层：把生成层/引用层里「让大模型当裁判打分」的调用收口。

生成层（Faithfulness / Answer Relevancy）与引用层（Citation Precision / Recall）
都借鉴 RAGAS 的思路：用另一个 LLM 当裁判，对「回答 + 资料 + 问题」给出结构化判定。
本模块提供：
  call_llm              底层文本调用（逐 token 拼接出整段）
  parse_json            从裁判输出里稳健抽 JSON（容忍 ```json 围栏）
  judge_faithfulness    回答是否只陈述资料支撑的事实
  judge_relevancy_questions  由回答反推它能回答的问题（供向量相似度算 Answer Relevancy）
  judge_citation        逐句判定引用是否支撑（供 Citation Precision / Recall）
  judge_fact_coverage   视觉解析文本是否覆盖了 gold 事实点（供视觉解析 Fact Recall）

裁判输出一律强制 JSON；解析失败时抛 ValueError，由上层记为失败样本并跳过，绝不捏造分数。
"""
import json
import re
from typing import Any, Dict, List

from app.core.llm import get_llm_client


async def call_llm(messages: List[Dict[str, str]]) -> str:
    """底层 LLM 文本调用：逐 token 拼接出整段文本。"""
    out = ""
    async for token in get_llm_client().stream_chat(messages):
        out += token
    return out.strip()


def parse_json(text: str):
    """从裁判输出里稳健抽出 JSON：先剥离 markdown 围栏，再整体解析，最后退化为截取首对括号。"""
    if not text:
        raise ValueError("裁判输出为空")
    cleaned = re.sub(r"```[a-zA-Z]*", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_ch)
        if start == -1:
            continue
        end = cleaned.rfind(close_ch)
        if end > start:
            return json.loads(cleaned[start:end + 1])
    raise ValueError(f"裁判输出无法解析为 JSON：{cleaned[:120]!r}")


_FAITHFULNESS_PROMPT = """你是严格的事实核查裁判。判断「回答」中的每个原子陈述是否能被「资料」支撑，用于检测回答有无编造（幻觉）。

请把回答拆成若干原子陈述，逐条判断是否被资料支撑。只输出 JSON 数组，不要输出任何其他文字：
[{{"statement": "陈述原文", "supported": true}}]   # supported 为 true 或 false

<资料>
{context}

<回答>
{answer}"""


async def judge_faithfulness(answer: str, context_text: str) -> Dict[str, Any]:
    """返回 {"supported": 被资料支撑的陈述数, "total": 陈述总数}。"""
    text = await call_llm([
        {"role": "user", "content": _FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)},
    ])
    claims = parse_json(text)
    if not isinstance(claims, list):
        raise ValueError(f"faithfulness 裁判返回的不是数组：{claims!r}")
    supported = 0
    total = 0
    for c in claims:
        if isinstance(c, dict) and c.get("statement") is not None:
            total += 1
            if c.get("supported"):
                supported += 1
    return {"supported": supported, "total": total}


_RELEVANCY_PROMPT = """请基于下面的回答，生成 3 个这个回答能够回答的问题。只输出 JSON 字符串数组，例如 ["问题1","问题2","问题3"]，不要输出任何其他文字。

<回答>
{answer}"""


async def judge_relevancy_questions(answer: str) -> List[str]:
    """由回答反推它能回答的问题，供 Answer Relevancy 计算向量相似度。"""
    text = await call_llm([
        {"role": "user", "content": _RELEVANCY_PROMPT.format(answer=answer)},
    ])
    qs = parse_json(text)
    if not isinstance(qs, list):
        raise ValueError(f"relevancy 裁判返回的不是数组：{qs!r}")
    return [str(q) for q in qs]


_CITATION_PROMPT = """请分析「回答」的引用质量。把回答按句拆分，对每一句给出三个字段：
- attributable：该句是否是需要事实依据支撑的陈述（寒暄/过渡句为 false）
- cited：该句引用了哪些资料编号（取句末 [资料N] 中的 N，无则空数组）
- supported：该句的引用是否真的支撑这句陈述

只输出 JSON，形如：
{{"statements":[{{"text":"...","attributable":true,"cited":[1],"supported":true}}]}}

<资料>
{context}

<回答>
{answer}"""


async def judge_citation(answer: str, context_text: str) -> List[Dict[str, Any]]:
    """返回逐句判定列表，每句 {text, attributable, cited, supported}。"""
    text = await call_llm([
        {"role": "user", "content": _CITATION_PROMPT.format(context=context_text, answer=answer)},
    ])
    data = parse_json(text)
    statements = data.get("statements") if isinstance(data, dict) else data
    if not isinstance(statements, list):
        raise ValueError(f"citation 裁判返回结构异常：{data!r}")
    return statements


_FACT_COVERAGE_PROMPT = """你是严格的信息抽取裁判。给定一段「图片解析文本」和若干「事实点」，逐条判断该事实点是否被解析文本覆盖。

判定标准：事实点的关键信息（文字内容、数值、名称）在解析文本中明确出现或可由原文直接推出，记 covered=true；解析文本完全未提及、或只提到相关主题但缺少该事实点的关键信息，记 covered=false。

只输出 JSON 数组，不要输出任何其他文字：
[{{"fact": "事实点原文", "covered": true}}]   # covered 为 true 或 false

<图片解析文本>
{parsed}

<事实点>
{facts}"""


async def judge_fact_coverage(parsed_text: str, facts: List[str]) -> List[Dict[str, Any]]:
    """逐条判定 gold 事实点是否被视觉解析文本覆盖，返回 [{fact, covered}]。

    覆盖数 / 事实点总数即该样本的 Fact Recall（由调用方聚合）。
    """
    facts_text = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
    text = await call_llm([
        {"role": "user", "content": _FACT_COVERAGE_PROMPT.format(parsed=parsed_text, facts=facts_text)},
    ])
    rows = parse_json(text)
    if not isinstance(rows, list):
        raise ValueError(f"fact_coverage 裁判返回的不是数组：{rows!r}")
    # 裁判可能漏判/多判：以 gold 事实点为准对齐，漏判记 covered=False（保守口径）
    by_fact = {
        str(r.get("fact", "")).strip(): bool(r.get("covered"))
        for r in rows if isinstance(r, dict)
    }
    return [
        {"fact": f, "covered": by_fact.get(f.strip(), False)}
        for f in facts
    ]
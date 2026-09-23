"""Agent 运行时：ReAct 工具循环（从单一 RAG 升级为 Agentic RAG 的核心）。

循环逻辑：
  1. 组装初始消息（提示词 + 记忆 + 问题；KB 资料不再前置注入，由模型自决检索）
  2. 流式请求 LLM（携带可用工具 schema）
       - token 事件 → 直接透传给前端
       - tool_calls → 执行工具 → 结果接续编号回灌 → 进入下一轮请求
  3. 防失控的三道刹车（分层，均为"软收尾"——答案始终由模型自己生成）：
       a. 重复调用拦截：同工具 + 同归一化 query 且上次拿到了非空结果时，
          再次调用不真正执行（向量检索对同词是确定性的，重跑只返回同一批资料），
          直接回灌拦截提示并引导模型换关键词改写；注意失败调用不入账，
          允许对瞬时故障用原词重试；
       b. 空结果熔断：连续 max_consecutive_empty 次检索为空/失败，
          提前收回工具（单次失败给重试机会，连续失败才止损），
          让模型基于已有信息坦诚收尾；
       c. 轮数硬上限：达到 search_max_rounds 后收回全部工具 schema，
          模型在协议层无法再发起 tool_call（不是提示词软约束），必然下一轮收尾。
     另在最后一次有工具的轮次结果后追加"最后机会"预告，让模型带着收尾意识检索。

分层约束：本模块只认识 LLM 抽象（core/llm）与工具注册表（tools/），
不依赖具体模型厂商与搜索引擎——换厂商改 .env，换引擎改 services/search。
"""
import json
import logging
import time
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.agent.prompts import build_system_messages
from app.agent.tools import TOOL_EXECUTORS, get_tools
from app.config import settings
from app.core.llm import get_llm_client
from app.services.memory import SessionMemory

# Agent 执行日志：服务端控制台可见完整决策轨迹（uvicorn 启动即输出）
logger = logging.getLogger("app.agent")


def _format_kb_block(i: int, r: Dict[str, Any]) -> str:
    """单条 KB 资料回灌模型时的格式（与历史 format_contexts 风格一致）。"""
    tag = " · FAQ" if r.get("collection") == "faq" else ""
    return f"【资料{i}】(来源: {r.get('source', '')}{tag})\n{r.get('text', '')}"


def _format_web_block(i: int, r: Dict[str, Any]) -> str:
    """单条网页资料回灌模型时的格式（web_results 带 title 字段）。"""
    return (
        f"[资料{i}] {r.get('title', '')}\n"
        f"来源：{r.get('source', '')}\n"
        f"{r.get('text', '')}"
    )


class AgentService:
    async def run_stream(
        self,
        question: str,
        memory: Optional[SessionMemory] = None,
        web_search_enabled: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """运行一轮问答，产出与 SSE 对应的事件流：
          {"type": "token", "content": str}            —— 回答文本增量
          {"type": "kb_searching", "query": str}       —— 正在检索本地知识库
          {"type": "searching", "query": str}          —— 正在联网搜索
          {"type": "sources", "documents": [...]}      —— KB 检索资料（编号已分配）
          {"type": "web_sources", "documents": [...]}  —— 网页资料（编号接续 KB 之后）
          {"type": "search_failed", ...}               —— 检索失败/空结果
        """
        messages = build_system_messages(question, memory)
        llm = get_llm_client()

        next_index = 1  # 全局编号计数器：KB 与网页资料共用一套编号，按到达顺序接续
        executed_rounds = 0
        llm_calls = 0   # 模型调用总次数（决策 + 生成）
        # 刹车 a：已执行过的 (工具名, 归一化 query)，重复出现直接拦截
        seen_calls: set = set()
        # 刹车 b：连续空结果/失败计数，命中即熔断；任一次拿到结果立即归零
        consecutive_empty = 0
        tools_revoked = False  # 熔断标记：一旦置位，后续轮次不再提供任何工具

        logger.info(
            "开始处理 | 问题=%r | 联网=%s",
            question[:40], web_search_enabled,
        )

        while True:
            # 轮数用尽（硬上限）或已被空结果熔断时，收回所有工具，
            # 模型在协议层无法发起 tool_call，只能基于已有信息收尾
            if (
                executed_rounds < settings.search_max_rounds
                and not tools_revoked
            ):
                tools = get_tools(web_search_enabled)
            else:
                tools = []
            tool_calls: List[Dict[str, Any]] = []

            # 每次循环 = 一次模型调用：要么决策调用工具，要么直接生成回答
            llm_calls += 1
            logger.info(
                "模型调用 #%d 开始 | 第%d轮 | 携带工具=%s",
                llm_calls, executed_rounds + 1, bool(tools),
            )
            yield {"type": "llm_start", "call": llm_calls, "round": executed_rounds + 1}

            async for event in llm.stream_chat_with_tools(
                messages, tools=tools or None
            ):
                if event["type"] == "token":
                    yield event
                else:  # tool_calls
                    tool_calls = event["calls"]

            if not tool_calls:
                break  # 模型已给出完整回答

            logger.info(
                "第%d轮 | 模型请求调用工具: %s",
                executed_rounds + 1,
                [c.get("function", {}).get("name") for c in tool_calls],
            )
            executed_rounds += 1
            # OpenAI 协议要求：带 tool_calls 的 assistant 消息先回传到历史
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": tool_calls}
            )

            for call in tool_calls:
                func = call.get("function", {})
                call_id = call.get("id", "")
                name = func.get("name", "")
                arguments = func.get("arguments", "{}")

                executor = TOOL_EXECUTORS.get(name)
                if executor is None:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": f"未知工具：{name}",
                        }
                    )
                    continue

                # 解析 query（kb_search / web_search 均用 query 参数）
                try:
                    query = str(json.loads(arguments or "{}").get("query") or "")
                except json.JSONDecodeError:
                    query = ""

                # 先发"检索中"事件：前端立即展示检索状态，同时把本次模型
                # 调用定型为"决策"。重复拦截也照发这套事件，保证日志节点闭合
                if name == "kb_search":
                    yield {"type": "kb_searching", "query": query, "round": executed_rounds}
                elif name == "web_search":
                    yield {"type": "searching", "query": query, "round": executed_rounds}

                # 刹车 a：同工具 + 同归一化关键词、且上次执行拿到过非空结果 →
                # 不真正执行。向量检索对同一 query 的结果是确定性的，重跑只会
                # 返回上方同一批资料；想补充信息必须改写关键词（不同 key 不拦）。
                # 注意：失败的调用不在账本里（见下方成功分支才入账），
                # 因此瞬时故障后用原词重试不会被误拦。
                norm_query = " ".join(query.strip().lower().split())
                dedup_key = (name, norm_query)
                if dedup_key in seen_calls:
                    logger.info(
                        "第%d轮 | 重复工具调用已拦截: %s(%r)",
                        executed_rounds, name, query[:40],
                    )
                    tool_content = (
                        f"【系统拦截】你已使用完全相同的关键词成功调用过 {name}。"
                        "检索结果对同一关键词是确定的，再次调用只会返回上方那批"
                        "相同资料，本次未重复执行。若现有资料不足以覆盖问题的某个"
                        "方面，请换一个角度或更换关键词重新构造检索词（禁止再使用"
                        "本条完全相同的关键词）；若判断无法通过检索补齐，请直接"
                        "基于已有资料作答或如实告知用户。"
                    )
                    yield {
                        "type": "search_failed",
                        "query": query,
                        "message": "重复检索已自动拦截（同工具、同关键词），请换关键词改写",
                        "round": executed_rounds,
                    }
                    messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": tool_content}
                    )
                    continue

                t0 = time.monotonic()

                result = await executor(arguments)

                # 分流取结果与格式化器：KB 走 sources 事件，网页走 web_sources 事件
                if name == "kb_search":
                    results = result.get("kb_results") or []
                    event_type = "sources"
                    fmt = _format_kb_block
                else:  # web_search
                    results = result.get("web_results") or []
                    event_type = "web_sources"
                    fmt = _format_web_block

                if results:
                    # 拿到有效结果：空结果 streak 归零，并把该词记入去重账本
                    # （失败不记账：瞬时故障后允许原词重试一次）
                    consecutive_empty = 0
                    seen_calls.add(dedup_key)
                    # 编号接续：从 next_index 起分配，KB 与网页共用一套编号
                    start = next_index
                    numbered: List[Dict[str, Any]] = []
                    lines: List[str] = []
                    for i, r in enumerate(results, start=start):
                        item = dict(r)
                        item["index"] = i
                        numbered.append(item)
                        lines.append(fmt(i, r))
                    next_index += len(numbered)
                    yield {"type": event_type, "documents": numbered, "round": executed_rounds}
                    tool_content = (
                        f"检索结果（引用时使用 [资料{start}] 起的编号）：\n\n"
                        + "\n\n".join(lines)
                    )
                else:
                    # 检索失败/空结果：执行器已备好可读的降级说明
                    tool_content = result.get("output", "检索失败")
                    # 轨迹事件：先发原始降级说明（前端失败日志保持干净）
                    yield {
                        "type": "search_failed",
                        "query": query,
                        "message": tool_content,
                        "round": executed_rounds,
                    }
                    consecutive_empty += 1
                    # 刹车 b：连续空结果达到阈值 → 熔断，下一轮起提前收回工具
                    if (
                        not tools_revoked
                        and consecutive_empty >= settings.max_consecutive_empty
                    ):
                        tools_revoked = True
                        tool_content += (
                            f"\n\n【系统提示】已连续 {consecutive_empty} 次检索无结果，"
                            "工具已提前收回，后续不再提供任何检索工具。"
                            "请基于已有资料作答；若确无相关信息，请如实告知用户。"
                        )
                        logger.warning(
                            "第%d轮 | 连续%d次检索为空，触发熔断，提前收回工具",
                            executed_rounds, consecutive_empty,
                        )

                # ---- 收尾预告：根据剩余轮数给模型下一步方向 ----
                # 熔断提示已含收尾指令，不再叠加
                if not tools_revoked:
                    remaining = settings.search_max_rounds - executed_rounds
                    if remaining == 1:
                        # 下一轮是最后一次有工具的请求
                        tool_content += (
                            "\n\n【系统提示】你还剩最后一轮工具调用机会。"
                            "若现有资料已足够回答，请直接作答；"
                            "仍需补充请精简关键词检索，之后工具将收回。"
                        )
                    elif remaining <= 0:
                        # 本轮工具执行完即达硬上限（search_max_rounds=1 也走这里）
                        tool_content += (
                            "\n\n【系统提示】工具调用轮数已达上限，下一轮不再提供"
                            "任何工具，请基于以上全部资料给出完整最终答案。"
                        )

                logger.info(
                    "工具 %s 完成 | 耗时=%.2fs | 命中=%d条",
                    name, time.monotonic() - t0, len(results),
                )

                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": tool_content}
                )


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService()

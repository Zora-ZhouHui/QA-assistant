"""多模态图→文适配器。

职责：把用户上传的图片按选定场景交给视觉模型解析成纯文本，
该文本再作为问题接入既有 kb_search + web_search 的 Agent 工具循环。
本模块只负责"图 → 文"，不碰任何检索逻辑；Agent 编排层也完全不感知图片来源。

设计取向：VisionParser 只逐段 yield 解析文本增量，不碰 SSE 事件——
"哪些事件落 trace、哪些只透传前端"这类 SSE 契约留在路由层，视觉模块不感知前端。
"""
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List

from app.config import settings
from app.core.llm import get_llm_client


@dataclass(frozen=True)
class SceneSpec:
    label: str          # 前端显示名（场景列表三选一的主文）
    desc: str           # 前端弹窗副标题（现在硬编码在 index.html，后续可由后端下发）
    parse_prompt: str   # 下发给视觉模型的解析指令
    followup: str       # 接入 RAG 链路时附加的追问句（驱动模型调用检索工具）
    # True = 视觉解析结果即最终答案，直接返回，不进入 Agent/KB/联网检索（如设计稿对比）
    direct_answer: bool = False


# 单一事实源：三份平行 dict 合并为一份 SceneSpec。
# 新增场景 = 在下面加一行；旧的 SCENE_PROMPTS / SCENE_LABELS / SCENE_FOLLOWUPS 已被移除。
SCENES: Dict[str, SceneSpec] = {
    "console_error": SceneSpec(
        label="控制台报错截图",
        desc="帮我解析 JS/TS 报错、错误栈、定位成因并给出修复建议",
        parse_prompt=(
            "你是前端报错解析助手，提取图中完整报错信息、错误栈、触发场景，"
            "用 MDN 标准技术术语输出纯文本，不要口语化描述"
        ),
        followup="请结合本地知识库与联网资料，定位该报错的成因并给出修复建议。",
    ),
    "layout_misalign": SceneSpec(
        label="布局错位截图",
        desc="识别布局方式、错位偏移、容器尺寸，给出修复方案",
        parse_prompt=(
            "你是前端布局解析助手，识别图中页面元素、容器尺寸、flex/grid 等布局方式、"
            "错位偏移量，输出标准 CSS 布局术语描述"
        ),
        followup="请结合本地知识库与联网资料，定位布局错位原因并给出修复方案。",
    ),
    "design_compare": SceneSpec(
        label="设计稿对比截图",
        desc="标注设计稿与页面的颜色/尺寸/间距差异，说明还原要点",
        parse_prompt=(
            "你是前端视觉还原助手，标注图中设计稿与页面的颜色、尺寸、间距差异，"
            "用前端可量化术语输出"
        ),
        followup="请结合本地知识库与联网资料，说明视觉还原要点与具体修改建议。",
        direct_answer=True,
    ),
}


class VisionParser:
    """图→文解析器。

    用法：
        async for chunk in parser.stream_parse(images, scene, note=""):
            # chunk 是解析文本的一个片段（逐 token/段 yield）
            ...
        # 解析完可通过 VisionParser.join() 拿全文，或直接拼接 yield 的片段
    """

    def __init__(self):
        self._llm = get_llm_client()

    async def stream_parse(
        self,
        images: List[str],
        scene: str,
        note: str = "",
    ) -> AsyncGenerator[str, None]:
        """逐段 yield 解析文本增量。

        校验与异常：
          - 未配视觉模型（settings.vision_model 为空）→ RuntimeError
          - scene 不在 SCENES 里 → ValueError（由路由转 error 事件）
          - 视觉模型返回空 → 本 generator 不 yield 任何片段（调用方需自行检测空结果）
          - 视觉模型调用异常 → 原样抛出（由路由捕获转 error 事件）
        """
        if not settings.vision_model:
            raise RuntimeError(
                "未配置视觉模型（VISION_MODEL），无法识别图片。"
            )
        if scene not in SCENES:
            raise ValueError(
                f"未知场景: {scene}（合法场景: {', '.join(SCENES.keys())}）"
            )

        spec = SCENES[scene]
        text = spec.parse_prompt
        if note:
            text = f"{text}\n\n用户补充说明：{note}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    *[{"type": "image_url", "image_url": {"url": u}} for u in images],
                ],
            }
        ]

        async for ev in self._llm.stream_chat_with_tools(
            messages,
            tools=None,
            model=settings.vision_model,
            base_url=settings.vision_base_url,
            api_key=settings.vision_api_key,
        ):
            if ev["type"] == "token":
                yield ev["content"]


def get_vision_parser() -> VisionParser:
    """工厂：与 get_agent_service / get_llm_client 同构，统一实例获取入口。"""
    return VisionParser()

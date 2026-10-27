"""全局配置：从环境变量 / .env 文件读取，所有可调参数集中在这里。"""
import json
from pathlib import Path
from typing import Dict

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = ""
    # OpenAI 兼容接口；换厂商只改 base_url / model 两个值。
    # 当前接阿里云百炼 qwen3.7-plus：模型 ID 必须小写，不能写成开源仓名 Qwen/Qwen3.7-Plus
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.7-plus"

    # 视觉（多模态）能力：用户上传截图后，先由视觉模型按场景指令把图解析成纯文本，
    # 该文本再接入 kb_search + web_search 的 RAG 链路。图片统一走 OpenAI 标准
    # image_url 格式（含 base64 dataURL），换厂商只改下面三项。
    # qwen3.7-plus 原生支持图文输入，故视觉能力复用同一模型与端点，无需另换模型。
    # vision_model 留空 = 未启用多模态，带图请求会被明确拦截提示（不静默降级）。
    vision_model: str = "qwen3.7-plus"
    vision_base_url: str = ""
    vision_api_key: str = ""

    # 单张图尺寸上限（前端已压缩，这里是双保险；按 dataURL 字符数计）
    max_image_bytes: int = 5 * 1024 * 1024

    # 模型单价表：JSON 字符串，格式 {"模型ID": {"input": 输入价, "output": 输出价}}，
    # 单位为 人民币元 / 1M tokens。接口不返回单价，须按厂商定价页自行维护；
    # 留空或缺某个模型时，该模型只统计 token 数、不显示成本（绝不捏造价格）。
    # 示例（数值以百炼定价页为准）：MODEL_PRICES={"qwen3.7-plus": {"input": 0.8, "output": 2.0}}
    model_prices: str = ""

    # 图片解析缓存：同图 + 同场景 + 同补充说明命中时直接返回缓存文本，
    # 跳过视觉模型调用（实测每张截图输入侧约数百至上千 tokens，命中即全省）。
    # 图片内容本身参与哈希，改图即换 key，不会返回旧图结果。
    vision_cache_enabled: bool = True

    # 本地 embedding 模型（首次运行时自动下载）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # HuggingFace 镜像站：国内网络直连 huggingface.co 常常超时，默认走镜像。
    # 能直连 huggingface.co 时，可在 .env 设 HF_ENDPOINT=https://huggingface.co 或留空。
    hf_endpoint: str = "https://hf-mirror.com"

    # RAG 参数
    chunk_size: int = 512       # 每个文本块的最大字符数
    chunk_overlap: int = 64     # 相邻块之间的重叠字符数
    top_k: int = 4              # 每次检索返回的文本块数量

    # 检索管线：dense=纯向量单路（默认，评测证明的最强路径）；hybrid=稠密 + BM25
    # 稀疏扩池（稀疏不改排序、只为给重排器更多候选）。见 app/services/retrieval/。
    # 注：重排关闭时 hybrid 与 dense 对 top_k 结果等价，故默认走 dense 避免无谓开销。
    retrieval_mode: str = "dense"    # dense | hybrid
    retrieval_candidate_k: int = 20  # 每路召回候选数（融合后的候选池大小）

    # 重排（Cross-Encoder）：对"全部候选池"做精排，代价是额外延迟。
    # 模型缺失/加载失败时自动降级为不重排（记日志），回退稠密序，不阻断检索。
    # 默认关闭：评测显示 bge-reranker-base 让 L4 Recall@1 从 84% 掉到 80.5%（负收益），
    # 更强的 v2-m3 在本机 CPU 上评测不可接受；重排作为可开关能力保留（见 README 评测段）。
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # 联网搜索（Tavily）：与 LLM 配置完全独立，换搜索引擎只改 SEARCH_PROVIDER
    search_provider: str = "tavily"
    tavily_api_key: str = ""            # 可选：不填不影响启动，触发联网时才报错降级
    search_max_results: int = 5         # 每次搜索返回的网页数
    search_depth: str = "advanced"      # basic=1 credit/次；advanced=2 credits，正文更全，可减少二次搜索
    search_country: str = "china"       # 地域限定（Tavily country 参数），留空则不限
    search_timeout: float = 15.0        # 搜索请求超时（秒）
    search_max_rounds: int = 3          # 单次问答最多几轮工具调用（最后安全网，硬上限）
    max_consecutive_empty: int = 2      # 连续几次检索为空/失败即熔断，提前收回工具
    web_search_default: bool = True     # 前端联网开关的默认值

    # 会话记忆压缩阈值（token 粗估，见 app/services/memory.py）
    # 超过 window 触发压缩，压到 target 以下为止；target < window 保证每轮压缩确实减负
    # 取值按 64K 级上下文留足安全余量；切换到更大上下文窗口的模型时可适当调大
    memory_window_tokens: int = 20000  # 近轮原文累计超过此值即触发滚动压缩
    memory_target_tokens: int = 10000  # 压缩后剩余原文回落到的目标值

    # 知识文件存放路径：把 HTML / TXT / Markdown 等文件放进这个目录，服务启动时自动扫描入库
    data_dir: Path = PROJECT_ROOT / "data"

    # FAQ 多数据源注册表：要增减 / 启停题库网站，直接编辑这个 JSON
    faq_sources_file: Path = PROJECT_ROOT / "faq_sources.json"

    # D2L（动手学深度学习）文档站：爬虫入口
    d2l_base_url: str = "https://zh.d2l.ai"

    @property
    def price_map(self) -> Dict[str, Dict[str, float]]:
        # 解析 MODEL_PRICES；非法 JSON / 非对象一律视为未配置（不阻断启动）
        if not self.model_prices:
            return {}
        try:
            data = json.loads(self.model_prices)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    @property
    def chroma_dir(self) -> Path:
        # Chroma 向量库持久化目录（重启后知识库仍在）
        return self.data_dir / "chroma"

    @property
    def sessions_dir(self) -> Path:
        # 会话记忆落盘目录：每个 session 一个子目录（messages.jsonl + summary.md）
        return self.data_dir / "sessions"

    @property
    def manifest_file(self) -> Path:
        # 索引清单：记录每个已入库文件的签名，支撑启动时的增量同步
        return self.data_dir / "index_manifest.json"

    @property
    def faq_manifest_file(self) -> Path:
        # FAQ 索引清单：记录每道题答案的签名，支撑 FAQ 的增量同步
        return self.data_dir / "faq_manifest.json"

    @property
    def faq_cache_dir(self) -> Path:
        # FAQ 爬取的 md 文件落盘缓存（离线重跑 / 调试时跳过网络）
        return self.data_dir / "faq_cache"

    @property
    def faqs_file(self) -> Path:
        # 洗数后的统一 FAQ 数据：启动时直接加载（不再实时抓取 GitHub）
        # 由 app.services.data_pipeline.wash_faqs 生成；
        # data/ 在 .gitignore 中但本文件单独提交
        return self.data_dir / "faqs.json"

    @property
    def d2l_cache_dir(self) -> Path:
        # D2L 文档爬取的 md 文件落盘目录（启动时自动入库）
        return self.data_dir / "d2l"

    @property
    def embedding_input_file(self) -> Path:
        # "分块之后、进入 embedding 模型之前"的数据快照（人工核对数据质量用）
        return self.data_dir / "embedding_input.json"

    @property
    def vision_cache_dir(self) -> Path:
        # 图片解析结果缓存目录（sha256 key → 解析文本）
        return self.data_dir / "vision_cache"


settings = Settings()

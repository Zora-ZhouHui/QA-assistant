"""全局配置：从环境变量 / .env 文件读取，所有可调参数集中在这里。"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM（OpenAI 兼容接口，当前接 GLM-4.7-Flash；切其他兼容模型只改 .env）
    llm_api_key: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    llm_model: str = "glm-4.7-flash"

    # 本地 embedding 模型（首次运行时自动下载）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # HuggingFace 镜像站：国内网络直连 huggingface.co 常常超时，默认走镜像。
    # 能直连 huggingface.co 时，可在 .env 设 HF_ENDPOINT=https://huggingface.co 或留空。
    hf_endpoint: str = "https://hf-mirror.com"

    # RAG 参数
    chunk_size: int = 512       # 每个文本块的最大字符数
    chunk_overlap: int = 64     # 相邻块之间的重叠字符数
    top_k: int = 4              # 每次检索返回的文本块数量

    # 会话记忆压缩阈值（token 粗估，见 app/services/memory.py）
    # 超过 window 触发压缩，压到 target 以下为止；target < window 保证每轮压缩确实减负
    # 当前按 GLM-4.7-Flash 200K 上下文设的保守值；切回 DeepSeek（64K）需相应调小
    memory_window_tokens: int = 20000  # 近轮原文累计超过此值即触发滚动压缩
    memory_target_tokens: int = 10000  # 压缩后剩余原文回落到的目标值

    # 知识文件存放路径：把 HTML / TXT / Markdown 等文件放进这个目录，服务启动时自动扫描入库
    data_dir: Path = PROJECT_ROOT / "data"

    # FAQ 多数据源注册表：要增减 / 启停题库网站，直接编辑这个 JSON
    faq_sources_file: Path = PROJECT_ROOT / "faq_sources.json"

    # D2L（动手学深度学习）文档站：爬虫入口
    d2l_base_url: str = "https://zh.d2l.ai"

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


settings = Settings()

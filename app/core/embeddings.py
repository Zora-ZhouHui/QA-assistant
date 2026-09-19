"""Embedding 封装：使用本地 bge 模型（sentence-transformers）。

模型在首次调用时才加载（懒加载），这样启动 Web 服务时不会立即占用内存，
也不会在还没安装依赖 / 下载模型时就报错。
"""
import os
from functools import lru_cache
from typing import List

from app.config import settings

# bge 中文模型官方建议：检索时给"问题"加这句指令，文档侧不加
_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    def __init__(self) -> None:
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            # 国内网络直连 huggingface.co 常常超时，默认走镜像站。
            # huggingface_hub 在 import 时读取 HF_ENDPOINT 环境变量，因此必须
            # 在 import sentence_transformers 之前设置；setdefault 让 shell 里
            # 显式 export 的 HF_ENDPOINT 优先级更高。
            if settings.hf_endpoint:
                os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
            # 懒导入：未装 sentence-transformers 时，不影响其他模块导入
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档切片向量化（批量）。"""
        model = self._ensure_loaded()
        vectors = model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """问题向量化（单条，带检索指令）。"""
        model = self._ensure_loaded()
        vector = model.encode(
            _QUERY_INSTRUCTION + text, normalize_embeddings=True
        )
        return vector.tolist()


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()

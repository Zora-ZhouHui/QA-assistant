"""重排：Cross-Encoder（sentence-transformers CrossEncoder）。

为什么需要重排：向量/BM25 都是"双塔"式的粗排，query 与 doc 各自编码后才交互，
精度有天花板；Cross-Encoder 把 (query, doc) 拼在一起过模型，精度显著更高。
代价是延迟（每个候选都要过一次模型），所以只对融合后的 top_n 精排。

模型缺失 / 加载失败 → 抛 RerankUnavailable，由 pipeline 捕获并降级（记日志，不阻断检索）。
"""
import logging
import math
import os
from functools import lru_cache
from typing import Any, Dict, List

from app.config import settings
from app.services.retrieval.base import Candidate

logger = logging.getLogger("app.retrieval")


class RerankUnavailable(Exception):
    """重排不可用（未装依赖 / 模型加载失败 / 推理异常）。"""


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _ensure_model(self):
        if self._model is None:
            # 与 embeddings 一致：国内直连 HF 常超时，import 前设置镜像端点
            if settings.hf_endpoint:
                os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
            try:
                from sentence_transformers import CrossEncoder
            except Exception as e:
                raise RerankUnavailable(f"未安装 sentence-transformers：{e}")
            try:
                self._model = CrossEncoder(self._model_name)
            except Exception as e:
                raise RerankUnavailable(f"加载重排模型 {self._model_name} 失败：{e}")
        return self._model

    def rerank(
        self, query: str, candidates: List[Candidate], top_k: int
    ) -> List[Candidate]:
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(query, c.get("text", "")) for c in candidates]
        try:
            raw = model.predict(pairs)
        except Exception as e:
            raise RerankUnavailable(f"重排推理失败：{e}")

        out: List[Candidate] = []
        for cand, s in zip(candidates, raw):
            try:
                value = float(s)
                prob = 1.0 / (1.0 + math.exp(-value))  # logit → 0~1 概率，供前端展示
            except Exception:
                value, prob = 0.0, 0.0
            item: Dict[str, Any] = dict(cand)
            item["rerank_score"] = value
            item["score"] = round(prob, 4)
            out.append(item)
        out.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return out[:top_k]


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(settings.rerank_model)

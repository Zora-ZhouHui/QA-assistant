"""Token 用量与成本计算。

口径约定：
  - token 数以接口返回的 usage 字段为准（账单口径，精确）；
    流式请求需开 stream_options.include_usage 才会在末个 chunk 返回 usage。
  - 单价接口不返回，由 config.model_prices 自行维护（模型 ID → 元/1M tokens）。
  - 成本 = 输入 tokens × 输入单价 + 输出 tokens × 输出单价。
  - 单价缺失时返回 None：宁可不显示成本，也不显示捏造的价格。
"""
from typing import Any, Dict, Optional

from app.config import settings


def compute_cost(model: str, usage: Dict[str, Any]) -> Optional[float]:
    """单次调用的成本（元）；模型未配单价时返回 None。"""
    price = settings.price_map.get(model)
    if not price:
        return None
    try:
        input_cost = usage.get("prompt_tokens", 0) * float(price.get("input", 0))
        output_cost = usage.get("completion_tokens", 0) * float(price.get("output", 0))
    except (TypeError, ValueError):
        return None
    return (input_cost + output_cost) / 1_000_000

"""视觉解析层离线评测：Fact Recall（LLM-as-a-Judge）。

评测对象是多模态"图→文"适配器（VisionParser.stream_parse）：给定截图与场景，
解析文本是否把回答该问题所需的关键事实点提取出来。指标：
  Fact Recall = 被解析文本覆盖的 gold 事实点数 / 事实点总数
聚合口径：overall 给 micro（全量事实点）与 macro（按样本平均），并按场景分组报告。

数据集：data/vision_eval.json，每条样本：
  {"id": "1f0c5cd1_0", "scene": "console_error",
   "image": "vision_eval/1f0c5cd1_0.jpeg",     # 相对 data/ 的路径
   "gold_facts": ["GET http://127.0.0.1:8000/favicon.ico", "404 (Not Found)"]}

评测期间临时关闭解析缓存（测的是模型解析能力，不是缓存命中）；
裁判解析失败的样本记 failed 并跳过，绝不捏造分数。
每次运行结果落盘 data/eval_runs/vision_<时间戳>.json。

用法（需 LLM_API_KEY 与 VISION_MODEL）：
    python -m app.eval.run_vision
"""
import argparse
import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.eval.judge import judge_fact_coverage
from app.services.vision import SCENES, get_vision_parser


def _to_data_url(path: Path) -> str:
    """图片文件 → base64 dataURL（与前端上传同构，直接进 VisionParser）。"""
    suffix = path.suffix.lstrip(".").lower() or "jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{suffix};base64,{b64}"


async def run() -> None:
    dataset_path = settings.data_dir / "vision_eval.json"
    if not dataset_path.exists():
        print("未找到 data/vision_eval.json，请先准备视觉评测集")
        return
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    print(f"视觉解析评测：{len(rows)} 个样本，逐条解析 + 事实覆盖判定…")

    # 测模型解析能力而非缓存命中：评测期间强制关缓存，跑完恢复
    cache_was = settings.vision_cache_enabled
    settings.vision_cache_enabled = False
    parser = get_vision_parser()

    samples: List[Dict[str, Any]] = []
    failed = 0
    try:
        for row in rows:
            sid = row.get("id", "?")
            scene = row.get("scene", "")
            facts = row.get("gold_facts") or []
            img_path = settings.data_dir / row.get("image", "")
            if scene not in SCENES or not facts or not img_path.exists():
                failed += 1
                print(f"  [skip] {sid:<16} 样本定义非法（场景/事实点/图片缺失）")
                continue

            parts: List[str] = []
            try:
                async for chunk in parser.stream_parse([_to_data_url(img_path)], scene):
                    parts.append(chunk)
                parsed = "".join(parts).strip()
                if not parsed:
                    raise ValueError("解析结果为空")
                judged = await judge_fact_coverage(parsed, facts)
            except Exception as e:
                failed += 1
                print(f"  [skip] {sid:<16} {e}")
                continue

            covered = sum(1 for j in judged if j["covered"])
            recall = covered / len(judged)
            samples.append(
                {
                    "id": sid,
                    "scene": scene,
                    "image": row["image"],
                    "parse_chars": len(parsed),
                    "facts": judged,
                    "covered": covered,
                    "total": len(judged),
                    "recall": round(recall, 4),
                }
            )
            print(f"  [ok ] {sid:<16} scene={scene:<15} recall={recall:.2f} ({covered}/{len(judged)})")
    finally:
        settings.vision_cache_enabled = cache_was

    if not samples:
        print("\n没有成功样本，无法聚合")
        return

    # micro：全量事实点口径；macro：按样本平均
    micro = sum(s["covered"] for s in samples) / sum(s["total"] for s in samples)
    macro = sum(s["recall"] for s in samples) / len(samples)

    per_scene: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        agg = per_scene.setdefault(s["scene"], {"samples": 0, "covered": 0, "total": 0})
        agg["samples"] += 1
        agg["covered"] += s["covered"]
        agg["total"] += s["total"]

    print("\n" + "=" * 60)
    print("视觉解析层离线评测结果（Fact Recall）")
    print("=" * 60)
    print(f"样本：{len(samples)} 成功 / {failed} 失败（跳过）")
    print(f"Fact Recall (micro) = {micro * 100:.2f}%")
    print(f"Fact Recall (macro) = {macro * 100:.2f}%")
    for scene, agg in per_scene.items():
        r = agg["covered"] / agg["total"]
        print(f"  {scene:<16} n={agg['samples']:<3} recall={r * 100:.2f}% ({agg['covered']}/{agg['total']})")

    # 落盘：供横向对比不同视觉模型 / 提示词版本的历次结果
    out_dir = settings.data_dir / "eval_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "vision_model": settings.vision_model,
                "samples": samples,
                "failed": failed,
                "micro": round(micro, 4),
                "macro": round(macro, 4),
                "per_scene": per_scene,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n详细结果已保存：{out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="视觉解析层评测（Fact Recall）")
    parser.parse_args()
    asyncio.run(run())


if __name__ == "__main__":
    main()

"""多模态 token 消耗量化测量（一次性脚本，结果写入 README）。

测量"图 → 文"适配器的真实 token 成本：
  A) 带图解析：场景 parse_prompt + 图片走视觉模型，读 usage（账单口径精确值）
  B) 同提示词纯文本：同一 parse_prompt 不带图，读 usage
  A - B 的 prompt_tokens 差值即图片本身的 token 占用。

用法：python3 scripts/measure_vision_tokens.py [图片路径...]
默认用 data/vision_eval/ 下前 3 张真实截图（来自用户实际上传的会话归档）。
"""
import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.llm import get_llm_client
from app.services.vision import SCENES

DEFAULT_IMAGES = sorted(Path("data/vision_eval").glob("*.jpeg"))[:3]


def to_data_url(path: Path) -> str:
    ext = path.suffix.lstrip(".") or "png"
    return f"data:image/{ext};base64," + base64.b64encode(path.read_bytes()).decode()


async def measure(image_path: Path):
    """对一张图分别跑带图/纯文本两次调用，返回 (带图usage, 纯文本usage, 解析输出字数)。"""
    llm = get_llm_client()
    prompt = SCENES["console_error"].parse_prompt

    async def run(with_image: bool):
        content = [{"type": "text", "text": prompt}]
        if with_image:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(image_path)}})
        usage, out = None, ""
        async for ev in llm.stream_chat_with_tools(
            [{"role": "user", "content": content}],
            tools=None,
            model=settings.vision_model,
            base_url=settings.vision_base_url,
            api_key=settings.vision_api_key,
        ):
            if ev["type"] == "token":
                out += ev["content"]
            elif ev["type"] == "usage":
                usage = ev["usage"]
        return usage, out

    usage_img, out_img = await run(True)
    usage_txt, _ = await run(False)
    return usage_img, usage_txt, len(out_img)


async def main():
    paths = [Path(p) for p in sys.argv[1:]] or DEFAULT_IMAGES
    print(f"视觉模型: {settings.vision_model} | 样本: {len(paths)} 张真实截图\n")
    print(f"{'图片':<24}{'带图入':>8}{'带图出':>8}{'纯文本入':>9}{'图片token':>10}")
    img_tokens = []
    for p in paths:
        if not p.exists():
            print(f"跳过不存在的文件: {p}")
            continue
        usage_img, usage_txt, out_len = await measure(p)
        delta = usage_img["prompt_tokens"] - usage_txt["prompt_tokens"]
        img_tokens.append(delta)
        print(
            f"{p.name:<24}{usage_img['prompt_tokens']:>8}{usage_img['completion_tokens']:>8}"
            f"{usage_txt['prompt_tokens']:>9}{delta:>10}  (解析输出 {out_len} 字)"
        )
    if img_tokens:
        avg = sum(img_tokens) / len(img_tokens)
        print(f"\n结论：每张截图平均占用 {avg:.0f} tokens（输入侧），"
              f"约为一条 20 字中文问题（~30 tokens）的 {avg/30:.0f} 倍")


asyncio.run(main())

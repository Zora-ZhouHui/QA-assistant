"""文本切片：把长文档切成适合向量检索的小块。

策略：先按段落（空行）聚合，尽量在段落边界切；
超过 chunk_size 的超长段落再按字符硬切，硬切块之间保留 overlap 重叠，
避免一句话被切断后上下文丢失。
"""
import re
from typing import List


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # 超长段落：先把缓冲 flush，再硬切
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_hard_split(para, chunk_size, overlap))
        elif len(buffer) + len(para) + 1 <= chunk_size:
            buffer = f"{buffer}\n{para}" if buffer else para
        else:
            chunks.append(buffer)
            buffer = para

    if buffer:
        chunks.append(buffer)
    return chunks


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    step = max(chunk_size - overlap, 1)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]

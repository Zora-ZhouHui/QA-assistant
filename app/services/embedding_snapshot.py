"""向量化前数据快照：把"分块之后、进入 embedding 模型之前"的数据落盘成 JSON，
方便在不启动服务、不加载 BGE 模型的情况下人工核对数据质量。

文件：data/embedding_input.json，分两段：
  - docs：知识文件（HTML/TXT/MD）解析后、split_text 切出的每个文本块，
          text 字段就是逐块送进 embedder.embed_documents 的字符串
  - faq ：爬取的 FAQ，只有 question 会被向量化（answer 只存 metadata，
          所以这里也只记录 question，快照内容与真实 embedder 输入保持一致）

增量维护方式与 index_manifest / faq_manifest 相同：
  - 文档/FAQ 重新入库 → 替换同 id 条目；删除 → 移除对应条目
  - 想看全量最新数据：python -m app.services.embedding_snapshot
    （只做解析/切片/爬取，不计算向量，也不读写 Chroma）
"""
import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger("embedding_snapshot")


class EmbeddingSnapshot:
    """JSON 快照存储：docs / faq 两个扁平数组，逐条对应一次 embedder 输入。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._docs: List[Dict[str, Any]] = []
        self._faq: List[Dict[str, Any]] = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._docs = data.get("docs", [])
            self._faq = data.get("faq", [])

    def _serialize(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "doc_chunk_count": len(self._docs),
            "faq_count": len(self._faq),
            "docs": self._docs,
            "faq": self._faq,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._serialize(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------- docs ----------
    def replace_doc(
        self, doc_id: str, source: str, chunks: List[str], persist: bool = True
    ) -> None:
        """文档（重新）入库时调用：先移除该文档旧切片，再写入新切片。"""
        self._docs = [d for d in self._docs if d["doc_id"] != doc_id]
        self._docs.extend(
            {
                "doc_id": doc_id,
                "source": source,
                "chunk_index": i,
                "char_count": len(text),
                "text": text,
            }
            for i, text in enumerate(chunks)
        )
        if persist:
            self.save()

    def remove_doc(self, doc_id: str, persist: bool = True) -> None:
        before = len(self._docs)
        self._docs = [d for d in self._docs if d["doc_id"] != doc_id]
        if persist and len(self._docs) != before:
            self.save()

    # ---------- faq ----------
    def replace_faq(
        self,
        faq_id: str,
        question: str,
        section: str,
        source_url: str,
        persist: bool = True,
    ) -> None:
        self._faq = [f for f in self._faq if f["faq_id"] != faq_id]
        self._faq.append(
            {
                "faq_id": faq_id,
                "section": section,
                "question": question,
                "source_url": source_url,
            }
        )
        if persist:
            self.save()

    def remove_faq(self, faq_id: str, persist: bool = True) -> None:
        before = len(self._faq)
        self._faq = [f for f in self._faq if f["faq_id"] != faq_id]
        if persist and len(self._faq) != before:
            self.save()

    def clear(self) -> None:
        self._docs = []
        self._faq = []


@lru_cache
def get_snapshot() -> EmbeddingSnapshot:
    return EmbeddingSnapshot(settings.embedding_input_file)


def rebuild_snapshot(output_path: Optional[Path] = None) -> Path:
    """重新生成整份快照：只解析/切片知识文件 + 从 JSON 加载 FAQ，不触碰 embedding 和 Chroma。

    直接复用入库流水线的同一套 loader / splitter / faq_loader，
    保证快照里看到的文本与真正向量化时完全一致。
    """
    # 函数内导入，避免 rag → snapshot → indexer → rag 的循环导入
    from app.services.document_loader import load_text
    from app.services.faq_loader import load_all_faqs
    from app.services.indexer import doc_id_for, scan_knowledge_files
    from app.services.splitter import split_text

    snapshot = get_snapshot()
    if output_path is not None:
        snapshot.path = output_path
    snapshot.clear()

    # 1. 知识文件：解析 → 切片（纯本地操作，很快）
    for path in scan_knowledge_files():
        rel = path.relative_to(settings.data_dir).as_posix()
        text = load_text(path)
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
        snapshot.replace_doc(doc_id_for(rel), rel, chunks, persist=False)

    # 2. FAQ：从洗数 JSON 加载，只记录被向量化的 question
    try:
        records = load_all_faqs()
    except Exception as e:
        logger.error("FAQ 加载失败，快照中仅包含文档切片：%s", e)
        records = []
    for rec in records:
        snapshot.replace_faq(
            rec.faq_id,
            question=rec.question,
            section=rec.section,
            source_url=rec.source_url,
            persist=False,
        )

    snapshot.save()
    logger.info(
        "向量化前快照已生成：%s（文档切片 %d 条，FAQ %d 条）",
        snapshot.path,
        len(snapshot._docs),
        len(snapshot._faq),
    )
    return snapshot.path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    path = rebuild_snapshot()
    print(f"快照已写入：{path}")

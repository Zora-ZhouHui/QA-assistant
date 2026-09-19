"""启动时自动建库：扫描 data/ 目录下的知识文件（HTML / TXT / Markdown），
解析 → 切片 → 向量化 → 写入 Chroma，让服务启动完成后知识库即刻可检索。

HTML 文件在这里只是"知识来源"——取出正文文字参与问答，
不会作为网页对外提供。

增量同步（为什么需要）：
  向量化比较耗时，没必要每次启动都把所有文件重算一遍。
  用 data/index_manifest.json 记录每个文件的"签名"（修改时间 + 大小）：
    - 新文件            → 入库
    - 签名变化的文件     → 删掉旧向量，重新入库
    - 目录里已删除的文件  → 删掉对应向量
    - 其余未变化的文件    → 跳过
  另外如果发现向量库是空的（比如手动删了 data/chroma/），清单作废，全部重建。
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.vectorstore import get_vector_store
from app.services.document_loader import SUPPORTED_EXTENSIONS
from app.services.embedding_snapshot import get_snapshot
from app.services.rag import get_rag_service

logger = logging.getLogger("knowledge_indexer")


def doc_id_for(rel_path: str) -> str:
    """由文件相对路径生成稳定的 doc_id：同一路径永远得到同一个 id，
    这样文件更新时可以精准定位并替换它的旧向量。"""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"qa-assistant.kb/{rel_path}").hex


def scan_knowledge_files() -> List[Path]:
    """递归扫描 data/ 下所有受支持的知识文件（跳过 chroma 持久化目录）。"""
    data_dir = settings.data_dir
    if not data_dir.exists():
        return []
    files: List[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        # chroma 目录里是向量库自己的落盘文件，不能当成知识文件
        if settings.chroma_dir in path.parents:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


class IndexManifest:
    """索引清单（data/index_manifest.json）：记录每个已入库文件的状态。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            self._entries = json.loads(path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, rel_path: str) -> Optional[Dict[str, Any]]:
        return self._entries.get(rel_path)

    def upsert(self, rel_path: str, **fields: Any) -> None:
        entry = self._entries.get(rel_path, {})
        entry.update(fields)
        entry["path"] = rel_path
        self._entries[rel_path] = entry
        self._save()

    def remove(self, rel_path: str) -> None:
        self._entries.pop(rel_path, None)
        self._save()

    def all_paths(self) -> List[str]:
        return list(self._entries.keys())

    def list_all(self) -> List[Dict[str, Any]]:
        """按路径排序返回全部条目，供接口展示。"""
        return [self._entries[k] for k in sorted(self._entries.keys())]


def load_manifest() -> IndexManifest:
    """读取索引清单（每次调用都读文件，保证看到启动同步后的最新状态）。"""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return IndexManifest(settings.manifest_file)


def sync_index() -> Dict[str, int]:
    """服务启动时调用：把 data/ 目录同步进向量库。

    返回各类操作的计数，用于日志输出。
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    store = get_vector_store()
    service = get_rag_service()

    # 扫描 data/ 下所有受支持的知识文件
    # 递归扫描所有子目录
    files = scan_knowledge_files()
    logger.info("在 data/ 下发现 %d 个知识文件，开始同步知识库…", len(files))

    # 向量库被清空过（例如手动删除了 data/chroma/）：清单失效，全部重建
    force_rebuild = store.count() == 0
    if force_rebuild and manifest.all_paths():
        logger.info("检测到向量库为空，将全部文件重新入库")

    indexed = updated = skipped = removed = failed = 0
    current_paths = set()

    for path in files:
        rel = path.relative_to(settings.data_dir).as_posix()
        current_paths.add(rel)
        stat = path.stat()
        old = None if force_rebuild else manifest.get(rel)

        # 签名一致：文件没动过，跳过重算
        if old and old.get("mtime_ns") == stat.st_mtime_ns and old.get("size") == stat.st_size:
            skipped += 1
            continue

        doc_id = doc_id_for(rel)
        try:
            if old:
                # 文件内容变了：先清掉旧向量，避免新旧片段混在一起
                store.delete(doc_id)
            result = service.ingest_file(path, source=rel, doc_id=doc_id)
            manifest.upsert(
                rel,
                doc_id=doc_id,
                source=rel,
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
                chunk_count=result["chunk_count"],
                indexed_at=datetime.now().isoformat(timespec="seconds"),
            )
            if old:
                updated += 1
                logger.info("[更新] %s → %d 个片段", rel, result["chunk_count"])
            else:
                indexed += 1
                logger.info("[新增] %s → %d 个片段", rel, result["chunk_count"])
        except Exception as e:
            # 单个文件失败不影响其他文件和服务启动
            failed += 1
            logger.error("[失败] %s：%s", rel, e)

    # 清理目录中已经不存在的文件
    for rel in manifest.all_paths():
        if rel not in current_paths:
            entry = manifest.get(rel)
            if entry:
                store.delete(entry["doc_id"])
                get_snapshot().remove_doc(entry["doc_id"])
            manifest.remove(rel)
            removed += 1
            logger.info("[移除] %s（文件已从 data/ 删除）", rel)

    logger.info(
        "知识库同步完成：新增 %d，更新 %d，跳过（未变化）%d，移除 %d，失败 %d",
        indexed,
        updated,
        skipped,
        removed,
        failed,
    )
    return {
        "indexed": indexed,
        "updated": updated,
        "skipped": skipped,
        "removed": removed,
        "failed": failed,
    }

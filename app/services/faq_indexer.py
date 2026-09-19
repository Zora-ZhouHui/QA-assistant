"""FAQ 启动同步：从洗数 JSON 加载题库 → 对齐答案签名 → 仅变化项重新向量化入库。

数据源是多个 GitHub 仓库，由 app.services.data_pipeline.wash_faqs 一次性
"洗"成统一格式的 JSON（data/faqs.json，按源分组的混合结构）。启动时直接从
JSON 加载，不再实时抓取 GitHub；源仓库有更新时，重跑洗数脚本刷新 JSON 即可。

同步逻辑沿用 indexer.py 的 manifest 模式：
  - 用 data/faq_manifest.json 记录每道题答案的 md5 签名
  - 签名一致跳过，避免每次启动都重算 embedding（bge 向量化是 CPU 密集操作）
  - 答案变了 → 删旧 + 重新入库
  - 题被删了 → 清掉对应向量
  - faq collection 为空但 manifest 非空（被手动清过） → 全部重建

按源隔离（多源之后必须如此）：
  - 某个源在 JSON 中缺失 / 条目格式错误只跳过它自己，
    绝不清掉它（或别的源）已入库的向量；
  - "源中删题"的清理只在 JSON 中存在的源内部做；
  - wash 时被移除的源，其存量向量原样保留。

注意：FAQ 只对 question 做 embedding，answer 仅进 metadata，参见 vectorstore.add_faq。
"""
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.core.vectorstore import get_vector_store
from app.services.embedding_snapshot import get_snapshot
from app.services.faq_loader import load_faqs_by_source
from app.services.rag import get_rag_service

logger = logging.getLogger("faq_indexer")

# 多源改造前 manifest 条目不带 source 字段；这些存量题全部来自这个源
_LEGACY_SOURCE = "agent-interview"


class FaqManifest:
    """FAQ 索引清单（data/faq_manifest.json）：记录每道题答案的签名。"""

    def __init__(self, path) -> None:
        import json

        self.path = path
        self._entries: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            self._entries = json.loads(path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        import json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, faq_id: str) -> Optional[Dict[str, Any]]:
        return self._entries.get(faq_id)

    def upsert(self, faq_id: str, **fields: Any) -> None:
        entry = self._entries.get(faq_id, {})
        entry.update(fields)
        entry["faq_id"] = faq_id
        self._entries[faq_id] = entry
        self._save()

    def remove(self, faq_id: str) -> None:
        if self._entries.pop(faq_id, None) is not None:
            self._save()

    def all_ids(self) -> List[str]:
        return list(self._entries.keys())

    def items(self) -> List[Tuple[str, Dict[str, Any]]]:
        """返回全部 (faq_id, entry)，供按源遍历清理。"""
        return list(self._entries.items())

    @staticmethod
    def entry_source(entry: Dict[str, Any]) -> str:
        """条目所属数据源；兼容没有 source 字段的旧清单。"""
        return entry.get("source") or _LEGACY_SOURCE


def _answer_signature(answer: str) -> str:
    return hashlib.md5(answer.encode("utf-8")).hexdigest()


def sync_faq() -> Dict[str, int]:
    """服务启动时调用：把各数据源的 FAQ 同步进向量库的 faq collection。

    返回各类操作的计数，用于日志输出。
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    manifest = FaqManifest(settings.faq_manifest_file)
    store = get_vector_store()
    service = get_rag_service()

    # 1. 从洗数 JSON 加载（失败的源只返回 CrawlFailure，不影响其他源）
    results = load_faqs_by_source()
    if not results:
        logger.warning("FAQ 数据文件为空或不可用，跳过同步")
    failed_sources = [
        name for name, res in results.items() if not isinstance(res, list)
    ]
    if failed_sources:
        logger.warning("以下数据源加载失败，本轮跳过：%s", ", ".join(failed_sources))

    # 2. 向量库被清空过（例如手动删除了 data/chroma/）：清单失效，全部重建
    force_rebuild = store.count_faq() == 0
    if force_rebuild and manifest.all_ids():
        logger.info("检测到 faq collection 为空，将全部 FAQ 重新入库")

    indexed = updated = skipped = removed = failed = 0

    # 3. 每个成功的源独立对齐：新增 / 更新 / 跳过 / 源内删除
    for source_name, records in results.items():
        if not isinstance(records, list):
            continue

        logger.info("[%s] 加载到 %d 道 FAQ，开始同步…", source_name, len(records))
        current_ids = set()

        for rec in records:
            current_ids.add(rec.faq_id)
            new_sig = _answer_signature(rec.answer)
            old = None if force_rebuild else manifest.get(rec.faq_id)

            # 签名一致：答案没动过，跳过重算
            if old and old.get("signature") == new_sig:
                skipped += 1
                continue

            try:
                # 答案变了或新增：先删旧向量（幂等），再重新入库
                store.delete_faq(rec.faq_id)
                service.ingest_faq(
                    faq_id=rec.faq_id,
                    question=rec.question,
                    answer=rec.answer,
                    source_url=rec.source_url,
                    section=rec.section,
                )
                manifest.upsert(
                    rec.faq_id,
                    source=rec.source,
                    source_url=rec.source_url,
                    locator=rec.locator,
                    section=rec.section,
                    signature=new_sig,
                    ingested_at=datetime.now().isoformat(timespec="seconds"),
                )
                if old:
                    updated += 1
                    logger.info("[更新][%s] %s", source_name, rec.faq_id)
                else:
                    indexed += 1
                    logger.info("[新增][%s] %s", source_name, rec.faq_id)
            except Exception as e:
                failed += 1
                logger.error("[失败][%s] %s：%s", source_name, rec.faq_id, e)

        # 4. 源内清理：只删"属于本成功源、但本轮题目表里已不存在"的题。
        #    失败源 / 已从注册表移除的源不在此处理，其向量保留。
        for faq_id, entry in manifest.items():
            if FaqManifest.entry_source(entry) != source_name:
                continue
            if faq_id in current_ids:
                continue
            store.delete_faq(faq_id)
            get_snapshot().remove_faq(faq_id)
            manifest.remove(faq_id)
            removed += 1
            logger.info("[移除][%s] %s（源中已删除）", source_name, faq_id)

    logger.info(
        "FAQ 同步完成：新增 %d，更新 %d，跳过（未变化）%d，移除 %d，失败 %d",
        indexed, updated, skipped, removed, failed,
    )
    return {
        "indexed": indexed,
        "updated": updated,
        "skipped": skipped,
        "removed": removed,
        "failed": failed,
    }

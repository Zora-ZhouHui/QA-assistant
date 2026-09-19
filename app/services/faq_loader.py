"""FAQ 数据加载器：从洗数生成的 JSON 读取 FAQ，替代启动时的在线抓取。

JSON 由 app.services.data_pipeline.wash_faqs（洗数脚本，该目录不随功能代码
发布）一次性生成，结构为按源分组的混合格式：
  {"agent-interview": [{...FaqRecord 字段...}, ...], "zero2agent": [...], ...}

本模块只依赖标准库与 app.config，不依赖 data_pipeline —— 即使数据采集代码
（爬虫 / 适配器 / 洗数脚本）不在仓库中，运行时入库链路也能凭 data/faqs.json
正常工作。FaqRecord / CrawlFailure 两个数据模型定义于此，被数据管线反向复用。

加载语义（与原 crawl_faqs_by_source 对齐，让 sync_faq 零改动切换数据源）：
  - 单个源在 JSON 中缺失 / 条目格式错误 → 该源记为 CrawlFailure，
    sync_faq 据此跳过该源（不清理其存量向量），与"在线抓取失败"的处理一致；
  - 文件整体缺失 / 解析失败 → 返回空字典，sync_faq 据此跳过同步；
  - 单条记录字段不完整 → 跳过该条，记 warning，不影响同源其他条目。
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from app.config import settings

logger = logging.getLogger("faq_loader")


@dataclass
class FaqRecord:
    """跨源统一的 FAQ 记录（数据采集端与运行时入库端共用的唯一模型）。

    faq_id     全局稳定主键：同源同题永远同一个 id（源前缀 + 源内序号），
               内容更新时靠它精准替换旧向量。
    source     数据源名字（faq_sources.json 里的 name），用于按源隔离同步。
    question   题目正文（唯一参与向量化的字段）。
    answer     答案正文（只存 metadata，检索后拼回给 LLM）。
    source_url 可回溯的原文地址（浏览器打开得到出处页面）。
    section    所属章节 / 分类（仅 metadata）。
    locator    源内定位信息（仓库相对路径 / JSON 中的 id），写入 manifest 便于排查。
    """

    faq_id: str
    source: str
    question: str
    answer: str
    source_url: str
    section: str = ""
    locator: str = ""


@dataclass
class CrawlFailure:
    """单个数据源抓取失败的标记（带错误信息），与成功的 List[FaqRecord] 互斥。

    历史上由在线抓取链路产生；现在加载预洗 JSON 时"整源缺失 / 格式错误"
    同样用它表达，保持 sync_faq 的按源隔离逻辑不变。
    """

    error: str


# 逐源结果：成功 → 记录列表；失败 → CrawlFailure
SourceResult = Union[List[FaqRecord], CrawlFailure]


def load_faqs_by_source(path: Optional[Path] = None) -> Dict[str, SourceResult]:
    """从 faqs.json 读取 FAQ，按源分组返回。

    返回类型与 crawl_faqs_by_source 完全一致，sync_faq 可直接替换调用。
    """
    if path is None:
        path = settings.faqs_file
    if not path.exists():
        logger.warning("FAQ 数据文件不存在：%s，跳过同步", path)
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("FAQ 数据文件解析失败：%s（%s）", path, e)
        return {}
    if not isinstance(raw, dict):
        logger.error("FAQ 数据文件顶层不是对象：%s", path)
        return {}

    results: Dict[str, SourceResult] = {}
    for source_name, items in raw.items():
        if not isinstance(items, list):
            results[source_name] = CrawlFailure(
                error=f"源 {source_name} 的条目不是列表"
            )
            continue
        records: List[FaqRecord] = []
        bad = 0
        for item in items:
            try:
                records.append(FaqRecord(**item))
            except Exception:
                bad += 1
        if bad:
            logger.warning("源 %s 跳过 %d 条格式异常的条目", source_name, bad)
        results[source_name] = records
    return results


def load_all_faqs(path: Optional[Path] = None) -> List[FaqRecord]:
    """拍平所有源的 FAQ（快照重建等场景使用）。"""
    records: List[FaqRecord] = []
    for result in load_faqs_by_source(path).values():
        if isinstance(result, list):
            records.extend(result)
    return records

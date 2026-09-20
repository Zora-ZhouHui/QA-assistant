"""搜索引擎适配器抽象。

与 faq_sources 同构的设计：
  本包只负责"去哪搜、怎么把结果归一化"，不关心模型是谁；
  agent 层的 web_search 工具调用本包，也不关心搜索引擎是谁。
换搜索引擎 = 写一个 SearchProvider 子类 + 在 registry 注册 + 改 .env。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class SearchError(Exception):
    """搜索失败（网络/配额/key 缺失等）。

    agent 层捕获后把错误文本回灌给模型，由模型降级为"基于自身知识作答
    并声明未能联网"，而不是让整轮问答崩掉。
    """


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """搜索并返回统一结构的结果列表。

        每条结果与向量检索结果同构（collection 字段区分来源），上游可统一处理：
          text:       喂给模型的正文摘要
          source:     网页 URL（前端渲染为可点击链接）
          title:      网页标题（前端卡片展示用）
          score:      搜索引擎的相关度评分（0~1，仅展示参考，与向量相似度不同义）
          collection: 固定 "web"
        """

"""FastAPI 应用入口。

启动方式（在项目根目录）：
  1. cp .env.example .env  并填入 LLM_API_KEY
  2. pip install -r requirements.txt
  3. 把 HTML / TXT / Markdown 等知识文件放进 data/ 目录
  4. uvicorn app.main:app --reload
  5. 浏览器打开 http://127.0.0.1:8000

服务启动时会自动扫描 data/ 目录，把知识文件解析、切片、向量化后写入 Chroma，
启动完成即可直接提问（见 app/services/indexer.py）。
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.api import routes_chat, routes_docs, routes_sessions
from app.config import settings
from app.services.faq_indexer import sync_faq
from app.services.indexer import sync_index

# D2L 爬虫属于数据采集管线（app/services/data_pipeline/，已在 .gitignore 中
# 忽略，不随功能代码发布）。采用可选导入：该目录缺失时应用照常启动，
# 仅跳过"在线爬取 D2L 文档"这一步（文档知识库暂时为空，不影响 FAQ 问答）。
try:
    from app.services.data_pipeline.d2l_crawler import crawl_all_pages
except ImportError:
    crawl_all_pages = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
# httpx 每次请求都会打 INFO，模型加载时刷屏，压到 WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)

WEB_STATIC_DIR = Path(__file__).parent / "web" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 会话记忆目录：每个 session 一个子目录，启动时确保存在
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)

    # 启动顺序：
    #   1) 爬取 D2L 文档站 → 落盘 md 到 data/d2l/
    #   2) 扫描 data/ 同步文档知识库（含刚爬取的 D2L md）
    #   3) 抓取 FAQ 题库同步到 faq collection
    # 都是 CPU/IO 密集的同步流程，放线程池避免阻塞事件循环。
    await run_in_threadpool(crawl_all_pages)
    await run_in_threadpool(sync_index)
    await run_in_threadpool(sync_faq)
    yield


app = FastAPI(title="知识问答助手", lifespan=lifespan)

app.include_router(routes_chat.router)
app.include_router(routes_docs.router)
app.include_router(routes_sessions.router)

# 前端静态资源（CSS / JS）
app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    """首页：聊天界面。"""
    return FileResponse(WEB_STATIC_DIR / "index.html")

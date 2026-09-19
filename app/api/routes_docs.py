"""知识库接口：只读查看已入库的知识文件。

知识文件不需要上传——把 HTML / TXT / Markdown 等文件放进 data/ 目录，
服务启动时会自动扫描并入库（见 app/services/indexer.py）。
"""
from fastapi import APIRouter

from app.services.indexer import load_manifest

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
def list_documents() -> dict:
    """知识文件列表（读索引清单，不依赖向量库）。"""
    return {"documents": load_manifest().list_all()}

"""文档解析：把不同格式的文件统一转换成纯文本（服务启动建库时调用）。

- TXT / Markdown：直接按文本读取
- HTML：BeautifulSoup 去掉标签和 script/style/nav，只留正文——
  HTML 在这里只是知识来源，不作为网页提供
"""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm"}


def load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".markdown"}:
        return _load_plain(path)
    if ext in {".html", ".htm"}:
        return _load_html(path)
    raise ValueError(f"不支持的文件格式: {ext}")


def _load_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    # separator 让块级元素之间保留换行，避免文字粘在一起
    return soup.get_text(separator="\n")

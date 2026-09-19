# 知识问答助手（RAG）

## 技术栈

| 分类        | 选型                           | 说明                                 |
| --------- | ---------------------------- | ---------------------------------- |
| 后端框架      | Python + FastAPI             | 异步支持好、自带接口文档（`/docs`）              |
| 大模型       | GLM-4.7-Flash（智谱）        | OpenAI 兼容接口，直接用 `openai` SDK 调用    |
| Embedding | `BAAI/bge-small-zh-v1.5`（本地） | sentence-transformers 加载，免费、数据不出本机 |
| 向量库       | Chroma（本地持久化模式）              | pip 安装即用，数据落盘在 `data/chroma/`      |
| 文档解析      | BeautifulSoup（HTML）          | TXT/Markdown 直接读取                  |
| 前端        | 原生 HTML/CSS/JS               | 无需 Node 构建，由 FastAPI 直接托管          |
| 流式输出      | SSE（Server-Sent Events）      | 回答逐 token 推送，边生成边显示                |

## 工作原理

**启动建库**（`app/services/indexer.py`，FastAPI lifespan 启动时执行一次）：

```
扫描 data/ 下的 HTML/TXT/MD 文件
  → 与 index_manifest.json 中记录的文件签名（修改时间+大小）比对
  → 新文件：解析 → 切片 → bge 向量化 → 写入 Chroma
  → 改动过的文件：删除旧向量后重新入库
  → 已删除的文件：清理对应向量
  → 未变化的文件：跳过（所以重启很快，不会重复向量化）
```

**问答链路**（`app/services/rag.py`）：

```
问题向量化 → Chroma 余弦相似度检索 top_k 片段
      → 片段编号拼进提示词 → GLM-4.7-Flash 流式生成 → SSE 推送到前端
```

提示词要求模型只根据检索到的资料回答、不足时明说"无法回答"，并在末尾标注引用来源，减少幻觉。

## 快速开始

```bash
# 1. 安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 放入知识文件
# 把 HTML（或 TXT/Markdown）文件放进 data/ 目录，支持子目录分类存放

# 3. 启动（启动时自动建库，首次会下载 bge 模型约 100MB）
# LLM_API_KEY 在 https://bigmodel.cn/ 控制台创建，每次启动前终端里 export 一下即可
export LLM_API_KEY="你的智谱密钥"
uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000> ，左侧可以看到已入库的知识文件，直接提问即可。
新增或修改知识文件后，重启服务即会自动增量更新索引。

接口文档（Swagger UI）：<http://127.0.0.1:8000/docs>

## FAQ 题库数据源

FAQ 题库来自多个 GitHub 仓库（注册表 `faq_sources.json`，适配器见 `app/services/faq_sources/`）。
源仓库更新后，重跑洗数脚本刷新统一数据文件 `data/faqs.json`，服务启动时直接从它加载，不再实时抓取：

```bash
python -m scripts.wash_faqs
```

若本机开着 Watt Toolkit（Steam++）等 GitHub 加速工具（会对 github.com 做本地 HTTPS 拦截，
Python 报 `CERTIFICATE_VERIFY_FAILED`），需把其 CA 拼进信任链再跑：

```bash
SSL_CERT_FILE=data/ca-bundle.pem python -m scripts.wash_faqs
```

新增同格式数据源只需在 `faq_sources.json` 加一段配置；新格式则写一个 `FaqSource` 子类并在
`app/services/faq_sources/registry.py` 注册。

## 可调参数（.env 或 config.py）

| 参数                 | 默认值                      | 说明                                 |
| ------------------ | ------------------------ | ---------------------------------- |
| `LLM_API_KEY`      | 无（必填）                    | 智谱 API 密钥                          |
| `LLM_MODEL`        | `glm-4.7-flash`           | 模型名                                |
| `EMBEDDING_MODEL`  | `BAAI/bge-small-zh-v1.5` | 本地 embedding 模型                    |
| `chunk_size`       | 512                      | 每个文本块最大字符数                         |
| `chunk_overlap`    | 64                       | 相邻块重叠字符数                           |
| `top_k`            | 4                        | 每次检索返回的片段数                         |
| `HF_ENDPOINT`      | 未设置                      | 设为 `https://hf-mirror.com` 可加速模型下载 |
| `MEMORY_WINDOW_TOKENS` | 20000                 | 近轮原文累计 token 超此值即触发滚动压缩           |
| `MEMORY_TARGET_TOKENS` | 10000                 | 压缩后剩余原文回落到的目标值                    |

## 已知限制

- 知识文件在启动时扫描，运行期间新增文件需要重启服务才会入库
- 无多轮对话记忆，每次提问独立
- 未做检索重排序（rerank），长文档场景检索精度有提升空间


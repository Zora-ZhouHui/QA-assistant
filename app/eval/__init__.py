"""Agent 评测模块（离线工具，与 runtime 服务解耦）。

从业界通用的「分层评估」出发，按成本从低到高拆成三层：
  检索层 —— Recall@K / Hit@K / MRR，零 LLM 成本（run_retrieval）；
  生成层 —— Faithfulness / Answer Relevancy，LLM-as-a-Judge（run_generation）；
  引用层 —— Citation Precision / Recall，LLM 判定（run_citation）。

三层共用 dataset（抽样）、metrics（纯函数聚合）、judge（LLM 裁判）等模块。
评测只消费业务模块（faq_loader / embeddings / vectorstore / llm），绝不反向耦合，
因此独立放在 app/eval/ 而不是塞进 services/。
"""
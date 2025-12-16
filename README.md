# IntentRecognitionDemo

LLM intent识别演示仓库，汇报主题：“从提示词到微调”。框架基于 LangChain (>1.0) 与 LangGraph (>1.0)，代码分三块：Prompt 工程、RAG、微调（后两部分现为占位，后续补充）。

## 项目结构
- `src/intent_recognition/prompting/`: Prompt 模板、可替换的提示库、可运行的 CLI。
- `src/intent_recognition/rag/`: RAG 管道与图编排（占位）。
- `src/intent_recognition/finetune/`: 微调数据准备与训练脚本（占位）。
- `src/intent_recognition/shared/`: 共享 schema/工具（后续加入）。
- `tests/`: 测试用例（后续补充）。
- `AGENTS.md`: 仓库贡献与演示指南。

## 快速开始
1) 创建并激活虚拟环境（已存在可跳过）：
```bash
python -m venv .venv && source .venv/bin/activate
```
2) 安装依赖：
```bash
pip install -r requirements.txt
```
3) 设置模型凭证（示例为 OpenAI）：
```bash
export OPENAI_API_KEY=sk-...
```

## Prompt 工程演示（意图分类）
运行 CLI，按需替换 prompt 或 labels：
```bash
python -m intent_recognition.prompting.cli --text "想查询订单 1234 的进度" \
  --prompt concise \
  --model gpt-4o-mini \
  --labels "greeting,order_status,complaint,product_query,fallback"
```
输出包含 `intent`、`confidence`、`rationale` 以及原始模型响应。若要切换提示风格，使用 `--prompt analysis_first` 或在 `prompts.py` 中新增模板并传递名称即可。

## 后续工作
- RAG：添加检索/向量库、图式编排（LangGraph）以及示例入口。
- 微调：准备意图分类数据集、训练脚本与评估。
- 测试：补充 pytest 用例与覆盖率报告。

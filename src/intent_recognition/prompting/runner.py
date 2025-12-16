"""
基于 Prompt 工程的意图分类运行工具。
"""

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .prompts import DEFAULT_LABELS, available_prompts, build_prompt_template


class IntentSchema(BaseModel):
    """定义输出 JSON 的字段与含义。"""

    intent: str = Field(..., description="预测的意图标签，只能在允许列表内")
    confidence: float = Field(..., description="0-1 置信度")
    rationale: str = Field(..., description="简短理由")


@dataclass
class IntentResult:
    intent: str
    confidence: float
    rationale: str
    raw: str


def build_chain(
    model: Optional[Runnable] = None,
    prompt_name: str = "concise",
    labels: Optional[List[str]] = None,
) -> Runnable:
    """
    组装可运行链路：prompt -> chat 模型 -> 字符串输出。

    - model: 任意支持聊天生成的 LangChain Runnable（默认 ChatOpenAI）。
    - prompt_name: prompt 注册表中的名称。
    - labels: 注入到 prompt 的标签列表。
    """
    parser = JsonOutputParser(pydantic_object=IntentSchema)
    prompt = build_prompt_template(
        prompt_name,
        labels,
        format_instructions=parser.get_format_instructions(),
    )
    if model is None:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 需要环境变量 OPENAI_API_KEY
    return prompt | model | parser


def classify_intent(
    text: str,
    model: Optional[Runnable] = None,
    prompt_name: str = "concise",
    labels: Optional[List[str]] = None,
) -> IntentResult:
    """运行链路并尝试解析 JSON 输出，失败则回退到 fallback。"""
    chain = build_chain(model=model, prompt_name=prompt_name, labels=labels or DEFAULT_LABELS)
    raw = chain.invoke({"text": text})
    try:
        parsed: IntentSchema = raw  # JsonOutputParser 返回 Pydantic 对象
        intent = parsed.intent
        confidence = float(parsed.confidence)
        rationale = parsed.rationale
        raw_text = parsed.model_dump_json()
    except Exception:
        intent, confidence, rationale, raw_text = "fallback", 0.0, "未能解析模型输出", str(raw)
    return IntentResult(intent=intent, confidence=confidence, rationale=rationale, raw=raw_text)


__all__ = [
    "IntentResult",
    "build_chain",
    "classify_intent",
    "available_prompts",
    "DEFAULT_LABELS",
]

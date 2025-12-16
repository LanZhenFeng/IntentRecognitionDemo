"""
基于 Prompt 工程的意图分类运行工具。
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .prompts import DEFAULT_LABELS, available_prompts, build_prompt_template


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
    prompt = build_prompt_template(prompt_name, labels)
    if model is None:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 需要环境变量 OPENAI_API_KEY
    return prompt | model | StrOutputParser()


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
        payload: Dict[str, object] = json.loads(raw)
        intent = str(payload.get("intent", "fallback"))
        confidence = float(payload.get("confidence", 0.0))
        rationale = str(payload.get("rationale", ""))
    except Exception:
        intent, confidence, rationale = "fallback", 0.0, "未能解析模型输出"
    return IntentResult(intent=intent, confidence=confidence, rationale=rationale, raw=raw)


__all__ = [
    "IntentResult",
    "build_chain",
    "classify_intent",
    "available_prompts",
    "DEFAULT_LABELS",
]

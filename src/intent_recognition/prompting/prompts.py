"""
Intent 分类提示库（支持随时替换）。

提示模板要保持简洁，可在演示时快速切换。
"""

from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate

DEFAULT_LABELS: List[str] = [
    "greeting",
    "order_status",
    "complaint",
    "product_query",
    "fallback",
]

# 提示模板注册表，键为可选名称。
PROMPT_VARIANTS: Dict[str, ChatPromptTemplate] = {
    "concise": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Classify the user text into one intent. Allowed intents: {labels}. "
                    "Return JSON with keys: intent (one of labels), confidence (0-1 float), "
                    "rationale (short). If unsure, use 'fallback'."
                ),
            ),
            ("human", "User text: {text}"),
        ]
    ),
    "analysis_first": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an intent classifier. Allowed intents: {labels}. "
                    "First think step-by-step, then answer with JSON keys intent/confidence/rationale. "
                    "If multiple intents match, pick the dominant one and explain briefly."
                ),
            ),
            ("human", "User text: {text}"),
        ]
    ),
}


def available_prompts() -> List[str]:
    """返回已注册的提示模板名称列表。"""
    return sorted(PROMPT_VARIANTS.keys())


def build_prompt_template(name: str, labels: List[str] | None = None) -> ChatPromptTemplate:
    """返回填充好标签的提示模板。"""
    labels = labels or DEFAULT_LABELS
    if name not in PROMPT_VARIANTS:
        raise KeyError(f"Unknown prompt '{name}'. Available: {', '.join(available_prompts())}")
    label_text = ", ".join(labels)
    return PROMPT_VARIANTS[name].partial(labels=label_text)

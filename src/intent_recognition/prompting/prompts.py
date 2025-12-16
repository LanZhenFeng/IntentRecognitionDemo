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
                    "你是意图分类器，只能输出一个意图。允许的意图集合：{labels}。"
                    "必须严格按照给定的 JSON 模式输出。若不确定，使用 'fallback'。"
                    "\n输出格式说明：{format_instructions}"
                ),
            ),
            ("human", "用户文本：{text}"),
        ]
    ),
    "analysis_first": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是意图分类器，先逐步思考，再给出 JSON 结果。允许的意图集合：{labels}。"
                    "必须严格符合提供的 JSON 模式，若多意图冲突，选主导意图并简短说明。"
                    "\n输出格式说明：{format_instructions}"
                ),
            ),
            ("human", "用户文本：{text}"),
        ]
    ),
}


def available_prompts() -> List[str]:
    """返回已注册的提示模板名称列表。"""
    return sorted(PROMPT_VARIANTS.keys())


def build_prompt_template(
    name: str,
    labels: List[str] | None = None,
    format_instructions: str = "",
) -> ChatPromptTemplate:
    """返回填充好标签与输出格式说明的提示模板。"""
    labels = labels or DEFAULT_LABELS
    if name not in PROMPT_VARIANTS:
        raise KeyError(f"Unknown prompt '{name}'. Available: {', '.join(available_prompts())}")
    label_text = ", ".join(labels)
    return PROMPT_VARIANTS[name].partial(labels=label_text, format_instructions=format_instructions)

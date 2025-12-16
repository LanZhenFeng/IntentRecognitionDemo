"""
Prompt 工程意图分类 CLI 入口。
"""

import json
import os
from typing import List, Optional

import typer
from langchain_openai import ChatOpenAI

from .prompts import DEFAULT_LABELS, available_prompts
from .runner import classify_intent

app = typer.Typer(add_completion=False)


def _load_labels(labels: Optional[str]) -> List[str]:
    if not labels:
        return DEFAULT_LABELS
    return [label.strip() for label in labels.split(",") if label.strip()]


@app.command()
def run(
    text: str = typer.Option(..., "--text", "-t", help="待分类的用户文本"),
    prompt_name: str = typer.Option(
        "concise", "--prompt", "-p", help=f"选择提示模板，可选：{', '.join(available_prompts())}"
    ),
    model_name: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="OpenAI 聊天模型名称（需 OPENAI_API_KEY）",
    ),
    labels: Optional[str] = typer.Option(
        None,
        "--labels",
        "-l",
        help="逗号分隔的标签列表，留空则使用默认标签",
    ),
    temperature: float = typer.Option(0.0, "--temperature", "-T", help="模型温度"),
):
    """
    使用可选提示模板将一条文本分类到意图标签。
    """
    if "OPENAI_API_KEY" not in os.environ:
        typer.echo("未设置 OPENAI_API_KEY；请先设置，或在代码中替换为其他模型。")
        raise typer.Exit(code=1)

    model = ChatOpenAI(model=model_name, temperature=temperature)
    label_list = _load_labels(labels)
    result = classify_intent(text=text, model=model, prompt_name=prompt_name, labels=label_list)
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


def main():
    app()


if __name__ == "__main__":
    main()

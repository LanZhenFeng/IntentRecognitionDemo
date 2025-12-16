# Repository Guidelines

## Architecture Overview
- Demo theme: LLM intent recognition “from prompt to fine-tune” using LangChain>1.0.0 and LangGraph>1.0.0.
- Core pillars (each must be swappable and independently demoable):
  - Prompting: intent classification via prompt engineering; prompts must be easily replaceable/configurable.
  - RAG: retrieval-enhanced intent detection; topology defined as a graph for clarity.
  - Fine-tune: supervised fine-tuning workflow for the same intent labels.

## Project Structure & Module Organization
- Suggested layout under `src/intent_recognition/`:
  - `prompting/`: prompt templates, prompt runner, prompt registry (e.g., YAML/JSON for easy swap).
  - `rag/`: retrievers, chunkers, vector store client, graph definitions.
  - `finetune/`: dataset prep, trainer, evaluation scripts; checkpoints excluded from git.
  - `shared/`: schemas (`IntentExample`, `IntentLabel`), evaluation metrics, logging.
- Tests mirror modules in `tests/` (`tests/prompting/test_classifier.py`, etc.); fixtures in `tests/fixtures/`.
- Notebooks in `notebooks/`; tiny anonymized samples in `data/`; large models/data stay external and are referenced only.

## Build, Test, and Development Commands
- Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Install deps (LangChain/LangGraph pinned >1.0): `pip install -r requirements.txt`.
- Run all tests: `pytest tests -q`.
- Format and lint: `ruff check src tests` and `black src tests`.
- Demo entry points (examples): `python -m intent_recognition.prompting.cli`, `python -m intent_recognition.rag.cli`, `python -m intent_recognition.finetune.cli`.

## Coding Style & Naming Conventions
- Python 3.11+; use type hints everywhere; enable `from __future__ import annotations` in new modules.
- Follow Black defaults (88 cols) and Ruff’s recommended rules; docstrings in Google style.
- Use descriptive module names (`prompt_runner.py`, `graph_builder.py`, `trainer.py`), `snake_case` for functions/variables, `PascalCase` for classes, and `SCREAMING_SNAKE_CASE` for constants.
- Keep functions short; extract helpers instead of deep nesting. Prefer pure functions for preprocessing logic.

## Testing Guidelines
- Tests mirror package structure and use `pytest`. Name tests `test_<unit>.py` and functions `test_<behavior>`.
- Include minimal fixtures and factories to avoid brittle tests; use parametrization over loops.
- Add regression tests for every bug fix; new features require at least one positive and one edge-case test (e.g., ambiguous intent, low-confidence retrieval).
- Aim for >85% coverage on new/modified modules; use `pytest --cov=src/intent_recognition --cov-report=term-missing`.

## Commit & Pull Request Guidelines
- Commits should be small, focused, and imperative (`Add intent matcher`, `Fix slot parsing edge cases`). Squash noisy WIP commits before review.
- PRs need a short summary, testing notes (`pytest -q`, lint/format commands), and any relevant screenshots or sample I/O.
- Link issues or tasks in the description. Highlight breaking changes and migration steps explicitly.
- Prefer draft PRs while work is in progress; request review only after tests and linters pass locally.

## Security & Configuration Tips
- Keep secrets out of the repo; use environment variables and `.env.example` to document required settings.
- Do not commit large datasets or models; store them externally and document retrieval steps.
- Review dependencies for licenses and vulnerabilities before adding them; pin versions in `requirements.txt`.

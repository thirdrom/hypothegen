"""
Узел planner(state) -> state.

Просит LLM (app/llm.py, мок-режим по умолчанию) разбить state["query"] +
state["constraints"] на 3-5 поисковых подзапросов. Ответ ожидается строго в
формате JSON {"subqueries": [...]}; здесь же он валидируется — не полагаемся
на то, что LLM вернёт ровно то, что просили, даже в реальном режиме.
"""

from __future__ import annotations

from app.llm import ALLOW_MOCK_FALLBACK, LLMError, call_llm_json
from app.state import State

SYSTEM_PROMPT = (
    "Ты — планировщик исследовательских подзапросов. "
    "Всегда отвечай СТРОГО в формате JSON без пояснений и markdown-обёртки: "
    '{"subqueries": ["...", "..."]}. От 3 до 5 подзапросов.'
)

MIN_SUBQUERIES = 3
MAX_SUBQUERIES = 5


def _build_prompt(query: str, constraints: dict) -> str:
    return (
        f"Основной запрос: {query}\n"
        f"Ограничения: {constraints or 'нет'}\n\n"
        "Сформулируй от 3 до 5 конкретных поисковых подзапросов, которые "
        "вместе покрывают основной запрос с учётом ограничений."
    )


def _mock_subqueries(query: str, constraints: dict) -> dict:
    """Детерминированный мок: подзапросы вокруг query, без сети и ключей."""
    base = query.strip().rstrip("?.")
    constraint_hint = ", ".join(map(str, constraints.keys())) if constraints else "без явных ограничений"
    return {
        "subqueries": [
            f"{base}: существующие методы и подходы",
            f"{base}: ограничения и риски ({constraint_hint})",
            f"{base}: аналогичные решения в смежных областях",
            f"{base}: количественные данные и результаты экспериментов",
        ]
    }


def planner(state: State) -> State:
    """Разбивает state["query"] + constraints на 3-5 подзапросов через LLM."""
    query = state["query"]
    constraints = state["constraints"]

    mock_response = _mock_subqueries(query, constraints)
    prompt = _build_prompt(query, constraints)

    try:
        response = call_llm_json(prompt, system=SYSTEM_PROMPT, mock_response=mock_response)
    except LLMError as exc:
        if not ALLOW_MOCK_FALLBACK:
            raise
        state["debate_log"].append(f"planner: ошибка LLM ({exc}), использую резервные подзапросы")
        response = mock_response

    subqueries = response.get("subqueries") if isinstance(response, dict) else None
    if not isinstance(subqueries, list) or not subqueries:
        state["debate_log"].append(
            "planner: LLM вернула пустой/некорректный subqueries, использую резервные"
        )
        subqueries = mock_response["subqueries"]

    # Нормализация: только непустые строки, без дублей, не больше MAX_SUBQUERIES.
    cleaned: list[str] = []
    for item in subqueries:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    subqueries = cleaned[:MAX_SUBQUERIES]

    # Контракт "3-5 подзапросов" держим даже если LLM вернула меньше 3.
    if len(subqueries) < MIN_SUBQUERIES:
        for fallback in mock_response["subqueries"]:
            if fallback not in subqueries:
                subqueries.append(fallback)
            if len(subqueries) >= MIN_SUBQUERIES:
                break

    state["subqueries"] = subqueries
    state["debate_log"].append(f"planner: сформировал {len(subqueries)} подзапрос(ов): {subqueries}")
    return state

"""
Оценка и ранжирование гипотез.

score_hypothesis(hypothesis, state) -> Hypothesis
    Через LLM (app/llm.py, мок по умолчанию) проставляет novelty/value/
    feasibility/risk/cost_of_error в [0,1] и rationale — строго на основе
    провенанс-полей самой гипотезы (и flagged_risks соответствующей Critique
    из state["critiques"]), а не "с потолка":
        novelty      <- novelty_justification, rejected_alternatives
        value        <- kpi_link
        feasibility  <- constraints_satisfied, conditions
        risk / cost_of_error <- validity_limits, flagged_risks критика

rank(hypotheses, weights=None) -> list[RankedHypothesis]
    score = wn*novelty + wv*value + wf*feasibility - wr*risk - we*cost_of_error
    Компоненты уже нормированы в [0,1] функцией score_hypothesis. Сортировка
    по убыванию score. Ранжируются только гипотезы, уже прошедшие provenance
    (фильтрация — забота вызывающего узла, см. app/nodes/ranker.py).
"""

from __future__ import annotations

from app.llm import ALLOW_MOCK_FALLBACK, LLMError, call_llm_json
from app.state import Critique, Hypothesis, RankedHypothesis, State

DEFAULT_WEIGHTS = {
    "novelty": 0.25,
    "value": 0.3,
    "feasibility": 0.2,
    "risk": 0.15,
    "cost_of_error": 0.1,
}

SCORE_SYSTEM_PROMPT = (
    "Ты — оценщик исследовательских гипотез. Оцени пять компонентов, каждый "
    "числом от 0 до 1, СТРОГО на основе указанных полей (не придумывай "
    "ничего вне них):\n"
    "  novelty       <- novelty_justification и rejected_alternatives "
    "(чем весомее обоснование новизны и чем больше содержательных "
    "альтернатив отклонено, тем выше)\n"
    "  value         <- kpi_link (чем прямее и конкретнее связь с целевым "
    "показателем, тем выше)\n"
    "  feasibility   <- constraints_satisfied и conditions (чем больше "
    "ограничений выполнено и чем реалистичнее режим, тем выше)\n"
    "  risk          <- validity_limits и flagged_risks (чем более серьёзны "
    "и многочисленны риски, тем ВЫШЕ risk)\n"
    "  cost_of_error <- validity_limits и flagged_risks (чем дороже ошибка, "
    "если гипотеза не сработает, тем выше)\n\n"
    "Отвечай СТРОГО в формате JSON без пояснений и markdown-обёртки:\n"
    '{"novelty": 0.0, "value": 0.0, "feasibility": 0.0, "risk": 0.0, '
    '"cost_of_error": 0.0, "rationale": "..."}\n'
    "rationale — 2-4 предложения, явно ссылающиеся на то, из какого поля "
    "гипотезы взята каждая оценка (это будет показано человеку при отборе)."
)


def _find_flagged_risks(hypothesis_id: str, critiques: list[Critique]) -> list[str]:
    """Возвращает flagged_risks критика, относящегося к данной гипотезе (или [])."""
    for critique in critiques:
        if critique.hypothesis_id == hypothesis_id:
            return critique.flagged_risks
    return []


def _build_score_prompt(hypothesis: Hypothesis, flagged_risks: list[str]) -> str:
    return (
        f"novelty_justification: {hypothesis.novelty_justification}\n"
        f"rejected_alternatives: {hypothesis.rejected_alternatives}\n\n"
        f"kpi_link: {hypothesis.kpi_link}\n\n"
        f"constraints_satisfied: {hypothesis.constraints_satisfied}\n"
        f"conditions: {hypothesis.conditions}\n\n"
        f"validity_limits: {hypothesis.validity_limits}\n"
        f"flagged_risks (от критика): {flagged_risks}"
    )


def _mock_score(hypothesis: Hypothesis, flagged_risks: list[str]) -> dict:
    """
    Детерминированная офлайн-оценка без сети/ключей. Каждый компонент
    вычисляется из конкретных провенанс-полей гипотезы (не рандом и не
    константа), так что разные гипотезы получают разные, воспроизводимые
    оценки, зависящие от содержания их полей.
    """
    novelty = min(len(hypothesis.novelty_justification) / 200, 1.0) * 0.6 + min(
        len(hypothesis.rejected_alternatives) / 3, 1.0
    ) * 0.4

    value = min(len(hypothesis.kpi_link) / 150, 1.0)

    if hypothesis.constraints_satisfied:
        satisfied = sum(1 for v in hypothesis.constraints_satisfied.values() if v is True)
        total = len(hypothesis.constraints_satisfied)
        feasibility = satisfied / total
    else:
        feasibility = 0.5  # ограничения не указаны -> нейтральная оценка

    risk = min(len(flagged_risks) / 3, 1.0)

    cost_of_error = min(len(hypothesis.assumptions) / 4, 1.0)

    novelty, value, feasibility, risk, cost_of_error = (
        round(min(max(x, 0.0), 1.0), 3) for x in (novelty, value, feasibility, risk, cost_of_error)
    )

    rationale = (
        f"novelty={novelty} — из novelty_justification "
        f"(\"{hypothesis.novelty_justification[:80]}\") и "
        f"{len(hypothesis.rejected_alternatives)} rejected_alternatives; "
        f"value={value} — из kpi_link (\"{hypothesis.kpi_link[:80]}\"); "
        f"feasibility={feasibility} — из constraints_satisfied "
        f"({hypothesis.constraints_satisfied}) и conditions ({hypothesis.conditions}); "
        f"risk={risk} и cost_of_error={cost_of_error} — из validity_limits "
        f"(\"{hypothesis.validity_limits[:80]}\") и {len(flagged_risks)} flagged_risks критика."
    )

    return {
        "novelty": novelty,
        "value": value,
        "feasibility": feasibility,
        "risk": risk,
        "cost_of_error": cost_of_error,
        "rationale": rationale,
    }


def score_hypothesis(hypothesis: Hypothesis, state: State) -> Hypothesis:
    """
    Оценивает одну гипотезу через LLM (мок по умолчанию) и записывает
    novelty/value/feasibility/risk/cost_of_error/rationale прямо в неё.

    Возвращает ту же (мутированную) Hypothesis, чтобы вызывающий код мог
    использовать её сразу в списковых выражениях.
    """
    flagged_risks = _find_flagged_risks(hypothesis.id, state["critiques"])
    mock_response = _mock_score(hypothesis, flagged_risks)
    prompt = _build_score_prompt(hypothesis, flagged_risks)

    try:
        response = call_llm_json(prompt, system=SCORE_SYSTEM_PROMPT, mock_response=mock_response)
    except LLMError as exc:
        if not ALLOW_MOCK_FALLBACK:
            raise
        state["debate_log"].append(
            f"ranker: ошибка LLM-оценки для {hypothesis.id} ({exc}), использую мок-оценку"
        )
        response = mock_response

    if not isinstance(response, dict):
        response = mock_response

    for field in ("novelty", "value", "feasibility", "risk", "cost_of_error"):
        raw_value = response.get(field)
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raw_value = mock_response[field]
        setattr(hypothesis, field, round(min(max(float(raw_value), 0.0), 1.0), 3))

    rationale = response.get("rationale")
    hypothesis.rationale = str(rationale) if rationale else mock_response["rationale"]

    return hypothesis


def rank(hypotheses: list[Hypothesis], weights: dict | None = None) -> list[RankedHypothesis]:
    """
    Считает взвешенный score для каждой гипотезы и сортирует по убыванию.

    score = wn*novelty + wv*value + wf*feasibility - wr*risk - we*cost_of_error

    Компоненты должны быть уже нормированы в [0,1] (см. score_hypothesis).
    weights частично переопределяет DEFAULT_WEIGHTS — можно передать только
    изменённые ключи.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    ranked = [
        RankedHypothesis(
            hypothesis=h,
            score=round(
                w["novelty"] * h.novelty
                + w["value"] * h.value
                + w["feasibility"] * h.feasibility
                - w["risk"] * h.risk
                - w["cost_of_error"] * h.cost_of_error,
                4,
            ),
        )
        for h in hypotheses
    ]
    ranked.sort(key=lambda rh: rh.score, reverse=True)
    return ranked

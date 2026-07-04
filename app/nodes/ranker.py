"""
Узел ranker(state) -> state.

Берёт state["hypotheses"], оставляет только те, у которых соответствующая
Critique в state["critiques"] имеет provenance_ok=True (гипотезы без
валидного провенанса в ранжирование не попадают вовсе — их отбраковал
critic ещё на предыдущем шаге). Для оставшихся вызывает score_hypothesis()
(LLM-оценщик, app/ranking.py) и затем rank() — результат кладёт в
state["ranked"], отсортированный по убыванию score.
"""

from __future__ import annotations

from app.ranking import DEFAULT_WEIGHTS, rank, score_hypothesis
from app.state import Critique, Hypothesis, State


def _provenance_ok_map(critiques: list[Critique]) -> dict[str, bool]:
    return {c.hypothesis_id: c.provenance_ok for c in critiques}


def ranker(state: State) -> State:
    """Оценивает и ранжирует гипотезы с provenance_ok=True; кладёт результат в state["ranked"]."""
    provenance_ok = _provenance_ok_map(state["critiques"])

    eligible: list[Hypothesis] = [h for h in state["hypotheses"] if provenance_ok.get(h.id) is True]
    skipped = len(state["hypotheses"]) - len(eligible)

    for hypothesis in eligible:
        score_hypothesis(hypothesis, state)

    weights = state["weights"] if state["weights"] else None
    ranked = rank(eligible, weights=weights)

    state["ranked"] = ranked
    used_weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    if ranked:
        top = ranked[0]
        state["debate_log"].append(
            f"ranker: проранжировано {len(ranked)} гипотез (пропущено без provenance_ok={skipped}), "
            f"weights={used_weights}. Топ: {top.hypothesis.id} score={top.score} — {top.hypothesis.rationale}"
        )
    else:
        state["debate_log"].append(
            f"ranker: нет гипотез с provenance_ok=True для ранжирования (пропущено={skipped})"
        )

    return state

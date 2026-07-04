"""
Узел critic(state) -> state.

Для каждой гипотезы из state["hypotheses"] делает два прохода:

  1. Провенанс — детерминированные правила, БЕЗ LLM. Если хоть одно правило
     нарушено, гипотеза сразу получает verdict="reject", provenance_ok=False.
     Нетрассируемую гипотезу не чинят, а отбраковывают — на LLM-ревью такая
     гипотеза не попадает вовсе.
  2. LLM-ревью физико-химической состоятельности и реалистичности conditions
     (только для гипотез с валидным провенансом, app/llm.py, мок по умолчанию)
     -> verdict "revise" с конкретными reasons при устранимых недочётах, или
     "accept".

Результат — Critique на каждую гипотезу, всё складывается в state["critiques"].

Про MAX_REVISIONS и место мутации state["iteration"]: решение "в какую ветку
идти" (generator или ranker) читает graph.py (route_after_critic) по чистому
чтению state["critiques"] — это и есть "reroute-логика в graph.py". Но сам
инкремент state["iteration"] обязан происходить здесь, внутри узла critic, а
не в condition-функции add_conditional_edges: LangGraph применяет только
изменения, возвращённые узлом как часть state, а мутации, сделанные внутри
routing-функции, в state следующего шага не попадают (см. комментарий в
graph.py и историю с зависшим циклом при первой реализации графа). Поэтому
здесь же критик принудительно ограничивает число ревизий: как только
state["iteration"] достигает MAX_REVISIONS, LLM-ревью больше не может
запросить "revise" — гипотеза принимается с пометкой в reasons, что лимит
ревизий исчерпан. Это гарантирует, что цикл critic -> generator завершается
не позднее MAX_REVISIONS итераций.
"""

from __future__ import annotations

from app.llm import ALLOW_MOCK_FALLBACK, LLMError, call_llm_json
from app.state import Critique, Evidence, Hypothesis, State

# Сколько раз LLM-ревью может запросить доработку, прежде чем критик
# принудительно примет гипотезу (не путать с provenance-reject — это разные
# оси: reject решается один раз и не подлежит ревизии, revise ограничен по
# числу попыток).
MAX_REVISIONS = 2

REVIEW_SYSTEM_PROMPT = (
    "Ты — критик-эксперт по физико-химической состоятельности инженерных "
    "гипотез. Оцени, реалистичен ли указанный режим (conditions) и логична "
    "ли связь между ним и заявленным эффектом. Отвечай СТРОГО в формате JSON "
    'без пояснений и markdown-обёртки: {"verdict": "accept" | "revise", '
    '"reasons": ["..."], "flagged_risks": ["..."]}. Если verdict="revise", '
    "reasons обязательны и конкретны (что именно поправить)."
)


def _step_references_evidence(step: str, evidence: list[Evidence]) -> bool:
    """
    Эвристика провенанса (без LLM): шаг рассуждения считается опирающимся на
    evidence, если явно упоминает его source_id, либо пересекается с текстом
    факта минимум тремя значимыми словами (длиннее 3 символов, без учёта
    регистра). Это допускает вольный пересказ факта, но не "рассуждение из
    ниоткуда".
    """
    step_lower = step.lower()
    for item in evidence:
        if item.source_id.lower() in step_lower:
            return True
        step_words = {w for w in step_lower.split() if len(w) > 3}
        fact_words = {w for w in item.fact.lower().split() if len(w) > 3}
        if len(step_words & fact_words) >= 3:
            return True
    return False


def _check_provenance(
    hypothesis: Hypothesis, valid_source_ids: set[str], constraints: dict
) -> tuple[bool, list[str]]:
    """
    Детерминированная проверка трассируемости гипотезы (без LLM).

    Возвращает (provenance_ok, список нарушений). Проверяются все правила
    независимо от первого найденного нарушения — чтобы Critique.reasons
    сразу показывал полную картину, а не только первую ошибку.
    """
    violations: list[str] = []

    if not hypothesis.evidence:
        violations.append("evidence пуст")
    else:
        bad_ids = sorted({e.source_id for e in hypothesis.evidence if e.source_id not in valid_source_ids})
        if bad_ids:
            violations.append(f"evidence ссылается на несуществующие source_id: {bad_ids}")

    if not hypothesis.reasoning_steps:
        violations.append("reasoning_steps пуст")
    else:
        unsupported = [
            i
            for i, step in enumerate(hypothesis.reasoning_steps)
            if not _step_references_evidence(step, hypothesis.evidence)
        ]
        if unsupported:
            violations.append(f"шаги reasoning_steps без опоры на evidence: {unsupported}")

    if not hypothesis.conditions:
        violations.append("conditions пуст")

    if not hypothesis.validity_limits.strip():
        violations.append("validity_limits пуст")

    contradicted = sorted(
        key
        for key, value in hypothesis.constraints_satisfied.items()
        if key in constraints and value is False
    )
    if contradicted:
        violations.append(f"constraints_satisfied противоречит заявленным constraints: {contradicted}")

    return (len(violations) == 0, violations)


def _mock_llm_review(hypothesis: Hypothesis, iteration: int) -> dict:
    """
    Детерминированный мок LLM-ревью, без сети и ключей.

    Пока iteration < MAX_REVISIONS, просит уточнить физико-химическое
    обоснование режима — это даёт предсказуемый и завершающийся цикл
    revise -> generator -> critic. С iteration >= MAX_REVISIONS всегда
    "accept" (дополнительно на всякий случай подстраховано принудительным
    клампом в critic(), см. ниже, — на случай реального LLM, который мог бы
    настаивать на revise бесконечно).
    """
    if iteration < MAX_REVISIONS:
        return {
            "verdict": "revise",
            "reasons": [
                f"Уточните физико-химическое обоснование режима {hypothesis.conditions} — "
                "из приведённых evidence не следует напрямую, почему именно эти значения "
                "дают заявленный эффект",
            ],
            "flagged_risks": [
                "Возможен неучтённый побочный эффект добавки/режима на другие свойства "
                "материала (пластичность, коррозионная стойкость)",
            ],
        }
    return {"verdict": "accept", "reasons": [], "flagged_risks": []}


def _build_review_prompt(hypothesis: Hypothesis) -> str:
    return (
        f"Гипотеза: {hypothesis.statement}\n"
        f"Режим (conditions): {hypothesis.conditions}\n"
        f"Предположения (assumptions): {hypothesis.assumptions}\n"
        f"Пределы применимости (validity_limits): {hypothesis.validity_limits}\n"
        f"Обоснование новизны: {hypothesis.novelty_justification}\n\n"
        "Оцени физико-химическую состоятельность и реалистичность указанного режима."
    )


def critic(state: State) -> State:
    """Проверяет провенанс и физико-химическую состоятельность каждой гипотезы."""
    valid_source_ids = {c.source_id for c in state["retrieved"]}
    constraints = state["constraints"]
    iteration = state["iteration"]

    critiques: list[Critique] = []
    for hypothesis in state["hypotheses"]:
        provenance_ok, violations = _check_provenance(hypothesis, valid_source_ids, constraints)

        if not provenance_ok:
            critiques.append(
                Critique(
                    hypothesis_id=hypothesis.id,
                    verdict="reject",
                    reasons=violations,
                    flagged_risks=[],
                    provenance_ok=False,
                )
            )
            continue

        mock_review = _mock_llm_review(hypothesis, iteration)
        prompt = _build_review_prompt(hypothesis)
        try:
            review = call_llm_json(prompt, system=REVIEW_SYSTEM_PROMPT, mock_response=mock_review)
        except LLMError as exc:
            if not ALLOW_MOCK_FALLBACK:
                raise
            state["debate_log"].append(
                f"critic: ошибка LLM-ревью для {hypothesis.id} ({exc}), использую мок"
            )
            review = mock_review

        verdict = review.get("verdict") if isinstance(review, dict) else None
        reasons = list(review.get("reasons") or []) if isinstance(review, dict) else []
        flagged_risks = review.get("flagged_risks") or [] if isinstance(review, dict) else []

        if verdict not in {"accept", "revise"}:
            verdict = "accept"  # провенанс валиден; странный ответ LLM не должен ронять прогон

        # Принудительный кламп: лимит ревизий исчерпан -> "revise" запрещён,
        # даже если LLM (реальная или мок) настаивает на доработке.
        if verdict == "revise" and iteration >= MAX_REVISIONS:
            reasons = reasons + ["Лимит ревизий исчерпан — гипотеза принята несмотря на замечания"]
            verdict = "accept"

        critiques.append(
            Critique(
                hypothesis_id=hypothesis.id,
                verdict=verdict,
                reasons=reasons,
                flagged_risks=flagged_risks,
                provenance_ok=True,
            )
        )

    # Инкремент здесь, а не в route_after_critic — см. docstring модуля.
    if any(c.verdict == "revise" for c in critiques):
        state["iteration"] += 1

    state["critiques"] = critiques
    n_reject = sum(1 for c in critiques if c.verdict == "reject")
    n_revise = sum(1 for c in critiques if c.verdict == "revise")
    n_accept = sum(1 for c in critiques if c.verdict == "accept")
    state["debate_log"].append(
        f"critic: {len(critiques)} гипотез проверено — accept={n_accept}, "
        f"revise={n_revise}, reject={n_reject} (iteration была {iteration}, стала {state['iteration']})"
    )
    state["critiques"] = critiques
    return state

"""
Узел generator(state) -> state.

LLM (app/llm.py, мок-режим по умолчанию) на основе state["retrieved"] +
state["external"] + query + constraints генерирует N=3 гипотезы строго под
модель Hypothesis (app/state.py). В промпт идёт только рантайм-контекст —
цель, ограничения, найденные фрагменты с их source_id — без текста ТЗ.

Промпт форсирует три группы полей Hypothesis:
  КАК сформулировано  -> derivation_method, evidence[], reasoning_steps[]
  ПОЧЕМУ именно это    -> kpi_link, novelty_justification, rejected_alternatives[]
  ПРИ КАКИХ УСЛОВИЯХ   -> conditions, assumptions[], constraints_satisfied, validity_limits

Каждая гипотеза, вернувшаяся от LLM (или мока), валидируется здесь же:
нарушение любого правила -> гипотеза отбрасывается и в state не попадает
(не "чинится" и не оставляется частично невалидной).
"""

from __future__ import annotations

import logging

from app.llm import ALLOW_MOCK_FALLBACK, LLMError, call_llm_json
from app.state import Chunk, Critique, Evidence, Hypothesis, Ref, State

logger = logging.getLogger("generator")

N_HYPOTHESES = 3
MAX_RETRIEVED_IN_PROMPT = 12  # чтобы не раздувать промпт, если retrieved большой
MAX_CHUNK_CHARS_IN_PROMPT = 400

DERIVATION_METHODS = ["analogy", "knowledge_gap", "counterfactual", "extrapolation", "combination"]

SYSTEM_PROMPT = (
    "Ты — генератор исследовательских гипотез. Тебе дан набор фрагментов "
    "источников (каждый со своим source_id) и внешних ссылок. Сформулируй "
    f"РОВНО {N_HYPOTHESES} гипотезы, каждая — строго на основе переданных "
    "фрагментов, без домыслов вне них.\n\n"
    "Отвечай СТРОГО в формате JSON без пояснений и markdown-обёртки:\n"
    '{"hypotheses": [ { ... }, { ... }, { ... } ]}\n\n'
    "Каждый объект гипотезы должен содержать поля:\n"
    '  "statement": краткая формулировка гипотезы (строка)\n'
    f'  "derivation_method": один из {DERIVATION_METHODS}\n'
    '  "evidence": список объектов {"source_id": "...", "fact": "...", "how_used": "..."}, '
    "source_id ОБЯЗАН быть одним из переданных ниже, минимум 1 элемент\n"
    '  "reasoning_steps": список строк — воспроизводимая цепочка вывода от факта к выводу, '
    "минимум 2 шага, каждый шаг опирается на конкретный evidence, без \"магии\"\n"
    '  "kpi_link": как гипотеза бьёт в целевой показатель из запроса (строка)\n'
    '  "novelty_justification": почему это не очевидное/известное решение (строка)\n'
    '  "rejected_alternatives": список отклонённых альтернатив с причиной отклонения, минимум 1\n'
    '  "conditions": объект с КОНКРЕТНЫМИ числовыми режимами (температура, %, атмосфера и т.п.), '
    "не общие слова\n"
    '  "assumptions": список строк — что должно быть истинным, чтобы гипотеза сработала\n'
    '  "constraints_satisfied": объект, отражающий соответствие переданным ограничениям\n'
    '  "validity_limits": КОНКРЕТНО, при каких значениях/режимах гипотеза перестаёт работать '
    "(failure modes), не общие слова\n\n"
    "Поля novelty/value/feasibility/risk/cost_of_error/rationale/id НЕ указывай — "
    "их проставит система отдельно."
)


def _extract_revision_reasons(critiques: list[Critique]) -> list[str]:
    """Извлекает reasons из критик с verdict="revise" (с предыдущего прохода critic)."""
    reasons: list[str] = []
    for critique in critiques:
        if critique.verdict == "revise":
            reasons.extend(critique.reasons)
    return reasons


def _format_feedback_for_prompt(revision_reasons: list[str]) -> str:
    if not revision_reasons:
        return ""
    bullets = "\n".join(f"- {reason}" for reason in revision_reasons)
    return (
        "\n\nЗамечания критика с предыдущей итерации (устрани их адресно в новой версии "
        f"гипотез):\n{bullets}\n"
    )


def _format_retrieved_for_prompt(retrieved: list[Chunk]) -> str:
    lines = []
    for chunk in retrieved[:MAX_RETRIEVED_IN_PROMPT]:
        text = chunk.text[:MAX_CHUNK_CHARS_IN_PROMPT].replace("\n", " ")
        lines.append(f"- [{chunk.source_id}] ({chunk.source}): {text}")
    return "\n".join(lines) if lines else "(источники не найдены)"


def _format_external_for_prompt(external: list[Ref]) -> str:
    lines = [f"- {ref.title} ({ref.year or '?'}): {ref.url}" for ref in external[:MAX_RETRIEVED_IN_PROMPT]]
    return "\n".join(lines) if lines else "(внешних ссылок нет)"


def _build_prompt(
    query: str, constraints: dict, retrieved: list[Chunk], external: list[Ref], revision_reasons: list[str]
) -> str:
    return (
        f"Цель: {query}\n"
        f"Ограничения: {constraints or 'нет явных ограничений'}\n\n"
        f"Фрагменты источников (используй ТОЛЬКО их для evidence):\n"
        f"{_format_retrieved_for_prompt(retrieved)}\n\n"
        f"Внешние ссылки (контекст, не для evidence.source_id):\n"
        f"{_format_external_for_prompt(external)}\n"
        f"{_format_feedback_for_prompt(revision_reasons)}\n"
        f"Сформулируй {N_HYPOTHESES} гипотезы по инструкции из системного промпта."
    )


def _mock_hypotheses(query: str, constraints: dict, retrieved: list[Chunk], revision_reasons: list[str]) -> dict:
    """
    Детерминированный мок без сети/ключей.

    Реально опирается на переданные retrieved: каждая гипотеза берёт evidence
    из существующих чанков (циклически, если их меньше N_HYPOTHESES), поэтому
    source_id всегда валиден относительно контекста конкретного прогона.
    Если retrieved пуст — гипотезы формулировать не на чем, возвращаем пустой
    список (валидатор ниже просто не добавит ни одной гипотезы в state).

    Если переданы revision_reasons (замечания критика с предыдущей
    итерации), они добавляются в assumptions как явно устранённые пункты —
    это делает адресную доработку видимой даже в мок-режиме.
    """
    if not retrieved:
        return {"hypotheses": []}

    hypotheses = []
    for i in range(N_HYPOTHESES):
        primary = retrieved[i % len(retrieved)]
        secondary = retrieved[(i + 1) % len(retrieved)]

        evidence = [
            {
                "source_id": primary.source_id,
                "fact": f"Фрагмент {primary.source_id} описывает: {primary.text[:120].strip()}",
                "how_used": "Взят как основной наблюдаемый факт для формулировки гипотезы",
            }
        ]
        if secondary.source_id != primary.source_id:
            evidence.append(
                {
                    "source_id": secondary.source_id,
                    "fact": f"Фрагмент {secondary.source_id} добавляет: {secondary.text[:120].strip()}",
                    "how_used": "Использован как подтверждающий/уточняющий факт",
                }
            )

        temperature = 800 + i * 25
        additive_pct = 1.5 + i * 0.5

        assumptions = [
            "Сырьё соответствует спецификации, указанной в источниках",
            "Оборудование допускает заданный температурный режим",
        ]
        if revision_reasons:
            assumptions.append(
                f"Учтено замечание критика с предыдущей итерации: {revision_reasons[0][:160]}"
            )

        hypotheses.append(
            {
                "statement": f"[MOCK #{i + 1}] Гипотеза по запросу «{query}» на основе {primary.source_id}",
                "derivation_method": DERIVATION_METHODS[i % len(DERIVATION_METHODS)],
                "evidence": evidence,
                "reasoning_steps": [
                    f"Из {primary.source_id} следует наблюдаемый факт (см. evidence[0]).",
                    f"Из {secondary.source_id} следует дополнительное подтверждение (см. evidence[-1])."
                    if secondary.source_id != primary.source_id
                    else f"Повторный анализ фрагмента {primary.source_id} с другого угла подтверждает вывод.",
                    f"Комбинируя факты из {primary.source_id} и {secondary.source_id}, формулируем "
                    f"гипотезу #{i + 1} и переносим её в режим с температурой {temperature}°C "
                    f"и добавкой {additive_pct}%.",
                ],
                "kpi_link": f"Прямо влияет на цель «{query}» через снижение затрат на процесс",
                "novelty_justification": (
                    f"Комбинация фактов из {primary.source_id} и {secondary.source_id} "
                    "не встречается вместе ни в одном отдельном источнике корпуса"
                ),
                "rejected_alternatives": [
                    "Использовать более дорогой референсный процесс без изменений — "
                    "отклонено из-за бюджетных ограничений",
                    f"Вариант #{i + 1} без изменения температуры — отклонён как менее эффективный",
                ],
                "conditions": {
                    "temperature_C": temperature,
                    "additive_pct": additive_pct,
                    "atmosphere": "N2" if i % 2 == 0 else "Ar",
                },
                "assumptions": assumptions,
                "constraints_satisfied": {str(k): True for k in constraints} if constraints else {},
                "validity_limits": (
                    f"Перестаёт работать при temperature_C > {temperature + 100} "
                    f"или additive_pct > {additive_pct + 3}%"
                ),
            }
        )
    return {"hypotheses": hypotheses}


def _validate_hypothesis(raw: dict, valid_source_ids: set[str], hyp_id: str) -> Hypothesis | None:
    """
    Проверяет одну гипотезу-кандидата по правилам задачи и либо возвращает
    валидную Hypothesis, либо None (гипотеза отбрасывается, не "чинится").
    """
    try:
        evidence_raw = raw.get("evidence") or []
        if not evidence_raw:
            logger.warning("Гипотеза %s отклонена: пустой evidence", hyp_id)
            return None

        evidence = [Evidence(**e) for e in evidence_raw]
        bad_source_ids = [e.source_id for e in evidence if e.source_id not in valid_source_ids]
        if bad_source_ids:
            logger.warning(
                "Гипотеза %s отклонена: evidence ссылается на несуществующие source_id %s",
                hyp_id,
                bad_source_ids,
            )
            return None

        reasoning_steps = raw.get("reasoning_steps") or []
        if len(reasoning_steps) < 2:
            logger.warning("Гипотеза %s отклонена: reasoning_steps короче 2 шагов", hyp_id)
            return None

        rejected_alternatives = raw.get("rejected_alternatives") or []
        if len(rejected_alternatives) < 1:
            logger.warning("Гипотеза %s отклонена: нет rejected_alternatives", hyp_id)
            return None

        conditions = raw.get("conditions") or {}
        if not conditions:
            logger.warning("Гипотеза %s отклонена: пустые conditions", hyp_id)
            return None

        validity_limits = (raw.get("validity_limits") or "").strip()
        if not validity_limits:
            logger.warning("Гипотеза %s отклонена: пустой validity_limits", hyp_id)
            return None

        hypothesis = Hypothesis(
            id=hyp_id,
            statement=raw["statement"],
            derivation_method=raw["derivation_method"],
            evidence=evidence,
            reasoning_steps=reasoning_steps,
            kpi_link=raw.get("kpi_link", ""),
            novelty_justification=raw.get("novelty_justification", ""),
            rejected_alternatives=rejected_alternatives,
            conditions=conditions,
            assumptions=raw.get("assumptions") or [],
            constraints_satisfied=raw.get("constraints_satisfied") or {},
            validity_limits=validity_limits,
            novelty=0.0,
            value=0.0,
            feasibility=0.0,
            risk=0.0,
            cost_of_error=0.0,
            rationale="",
        )
        return hypothesis
    except Exception as exc:  # некорректная структура/типы полей от LLM
        logger.warning("Гипотеза %s отклонена: не прошла схему Hypothesis (%s)", hyp_id, exc)
        return None


def generator(state: State) -> State:
    """Генерирует N_HYPOTHESES валидных Hypothesis и кладёт их в state["hypotheses"]."""
    query = state["query"]
    constraints = state["constraints"]
    retrieved = state["retrieved"]
    external = state["external"]

    # Если это повторный проход после critic (verdict="revise"), забираем его
    # reasons — доработка должна быть адресной, а не "generate заново вслепую".
    revision_reasons = _extract_revision_reasons(state["critiques"])

    valid_source_ids = {c.source_id for c in retrieved}
    mock_response = _mock_hypotheses(query, constraints, retrieved, revision_reasons)
    prompt = _build_prompt(query, constraints, retrieved, external, revision_reasons)

    try:
        response = call_llm_json(prompt, system=SYSTEM_PROMPT, mock_response=mock_response)
    except LLMError as exc:
        if not ALLOW_MOCK_FALLBACK:
            raise
        state["debate_log"].append(f"generator: ошибка LLM ({exc}), использую резервные гипотезы")
        response = mock_response

    raw_hypotheses = response.get("hypotheses") if isinstance(response, dict) else None
    if not isinstance(raw_hypotheses, list):
        raw_hypotheses = []

    validated: list[Hypothesis] = []
    for idx, raw in enumerate(raw_hypotheses):
        hyp_id = f"h{state['iteration']}_{idx}"
        hypothesis = _validate_hypothesis(raw, valid_source_ids, hyp_id)
        if hypothesis is not None:
            validated.append(hypothesis)

    state["hypotheses"] = validated
    state["debate_log"].append(
        f"generator: сгенерировал {len(validated)}/{len(raw_hypotheses)} валидных гипотез "
        f"(iteration={state['iteration']}, retrieved={len(retrieved)}, external={len(external)}, "
        f"учтено замечаний критика={len(revision_reasons)})"
    )
    return state

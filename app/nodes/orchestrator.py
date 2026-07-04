"""
Узел orchestrator(state) -> state.

Для каждой одобренной гипотезы (state["approved"] — список Hypothesis.id,
проставленный человеком через HITL) LLM (app/llm.py, мок по умолчанию)
формирует пошаговый протокол проверки: этапы, необходимые ресурсы, критерии
успеха и провала. В промпт идёт только рантайм-контекст самой гипотезы
(statement, conditions, assumptions, kpi_link, validity_limits) — без текста
ТЗ. Результат — Roadmap на каждую одобренную гипотезу, складывается в
state["roadmap"] списком.
"""

from __future__ import annotations

import logging

from app.llm import ALLOW_MOCK_FALLBACK, LLMError, call_llm_json
from app.state import Hypothesis, Roadmap, RoadmapStep, State

logger = logging.getLogger("orchestrator")

SYSTEM_PROMPT = (
    "Ты — оркестратор экспериментальной проверки инженерных гипотез. По "
    "гипотезе построй пошаговый протокол её практической проверки.\n\n"
    "Отвечай СТРОГО в формате JSON без пояснений и markdown-обёртки:\n"
    '{"steps": [ { ... }, { ... } ]}\n\n'
    "Каждый шаг должен содержать поля:\n"
    '  "name": короткое название этапа (строка)\n'
    '  "resources": список конкретных необходимых ресурсов (сырьё, оборудование, '
    "время, люди — строки)\n"
    '  "success_criteria": список измеримых критериев успеха этого этапа\n'
    '  "failure_criteria": список измеримых критериев провала этого этапа\n\n'
    "Протокол должен быть воспроизводим и опираться на conditions/validity_limits "
    "гипотезы — не общие фразы вроде «провести испытания», а конкретные шаги "
    "с числами и режимами из гипотезы."
)


def _build_prompt(hypothesis: Hypothesis) -> str:
    return (
        f"Гипотеза: {hypothesis.statement}\n"
        f"Режим (conditions): {hypothesis.conditions}\n"
        f"Предположения (assumptions): {hypothesis.assumptions}\n"
        f"Связь с KPI: {hypothesis.kpi_link}\n"
        f"Пределы применимости (validity_limits): {hypothesis.validity_limits}\n\n"
        "Построй пошаговый протокол проверки этой гипотезы по инструкции из "
        "системного промпта."
    )


def _mock_roadmap(hypothesis: Hypothesis) -> dict:
    """
    Детерминированный мок без сети/ключей. Шаги реально построены на полях
    гипотезы (conditions, kpi_link, validity_limits), а не абстрактны —
    разные гипотезы дают разные шаги.
    """
    conditions_str = ", ".join(f"{k}={v}" for k, v in hypothesis.conditions.items()) or "без заданных параметров"

    return {
        "steps": [
            {
                "name": "Подготовка сырья и оборудования",
                "resources": [
                    "сырьё, соответствующее спецификации источников",
                    f"оборудование, допускающее режим: {conditions_str}",
                ],
                "success_criteria": [
                    "Партия сырья прошла входной контроль без отклонений",
                    "Оборудование откалибровано под заданный режим",
                ],
                "failure_criteria": [
                    "Сырьё не проходит входной контроль",
                    "Оборудование не может стабильно держать режим",
                ],
            },
            {
                "name": f"Пилотный прогон в режиме {conditions_str}",
                "resources": [
                    "1 пилотная партия",
                    "измерительное оборудование для целевого показателя",
                ],
                "success_criteria": [
                    f"Пилотная партия выдержана в режиме {conditions_str} без отклонений",
                    "Измеренный эффект соответствует направлению, заявленному в гипотезе",
                ],
                "failure_criteria": [
                    "Режим не удаётся выдержать в заданных границах",
                    "Эффект отсутствует или противоположен ожидаемому",
                ],
            },
            {
                "name": "Проверка соответствия целевому KPI",
                "resources": ["данные пилотного прогона", "референсные показатели до изменения"],
                "success_criteria": [
                    f"Результат подтверждает связь с KPI: {hypothesis.kpi_link}"[:200],
                ],
                "failure_criteria": [
                    "Улучшение показателя не подтверждено статистически на пилотной партии",
                ],
            },
            {
                "name": "Проверка пределов применимости",
                "resources": ["расширенная серия испытаний за пределами номинального режима"],
                "success_criteria": [
                    "Поведение системы за пределами номинального режима соответствует "
                    f"заявленным failure modes: {hypothesis.validity_limits}"[:200],
                ],
                "failure_criteria": [
                    "Отказ наступает раньше или в других условиях, чем указано в validity_limits",
                ],
            },
        ]
    }


def _parse_roadmap(raw: dict, hypothesis_id: str) -> Roadmap:
    """
    Парсит ответ LLM/мока в Roadmap. Шаги с некорректной структурой
    отбрасываются по отдельности (не рушат весь протокол), но должны быть
    полными по смыслу: пустое имя или отсутствие критериев успеха/провала —
    повод пропустить шаг, а не выдумывать содержимое за LLM.
    """
    steps: list[RoadmapStep] = []
    for raw_step in raw.get("steps") or []:
        try:
            name = str(raw_step["name"]).strip()
            success_criteria = raw_step.get("success_criteria") or []
            failure_criteria = raw_step.get("failure_criteria") or []
            if not name or not success_criteria or not failure_criteria:
                logger.warning(
                    "Шаг протокола для %s отброшен: не хватает name/success_criteria/failure_criteria",
                    hypothesis_id,
                )
                continue
            steps.append(
                RoadmapStep(
                    name=name,
                    resources=raw_step.get("resources") or [],
                    success_criteria=success_criteria,
                    failure_criteria=failure_criteria,
                )
            )
        except Exception as exc:
            logger.warning("Шаг протокола для %s отброшен: %s", hypothesis_id, exc)
            continue
    return Roadmap(hypothesis_id=hypothesis_id, steps=steps)


def orchestrator(state: State) -> State:
    """Строит протокол проверки для каждой одобренной гипотезы -> state["roadmap"]."""
    hypotheses_by_id = {h.id: h for h in state["hypotheses"]}

    roadmaps: list[Roadmap] = []
    for hyp_id in state["approved"]:
        hypothesis = hypotheses_by_id.get(hyp_id)
        if hypothesis is None:
            state["debate_log"].append(
                f"orchestrator: гипотеза {hyp_id} из approved не найдена среди текущих hypotheses, пропущена"
            )
            continue

        mock_response = _mock_roadmap(hypothesis)
        prompt = _build_prompt(hypothesis)
        try:
            response = call_llm_json(prompt, system=SYSTEM_PROMPT, mock_response=mock_response)
        except LLMError as exc:
            if not ALLOW_MOCK_FALLBACK:
                raise
            state["debate_log"].append(
                f"orchestrator: ошибка LLM для {hyp_id} ({exc}), использую резервный протокол"
            )
            response = mock_response

        if not isinstance(response, dict):
            response = mock_response

        roadmap = _parse_roadmap(response, hyp_id)
        if not roadmap.steps:
            state["debate_log"].append(
                f"orchestrator: для {hyp_id} не получилось построить ни одного валидного шага"
            )
            continue

        roadmaps.append(roadmap)

    state["roadmap"] = roadmaps
    state["debate_log"].append(
        f"orchestrator: построено {len(roadmaps)}/{len(state['approved'])} протоколов проверки "
        f"(approved={state['approved']})"
    )
    return state

"""
Единая точка вызова LLM для всего графа.

call_llm_json(prompt, system=None, mock_response=None) -> dict

Просит модель вернуть строго JSON и парсит его. По умолчанию (USE_REAL=False)
к сети/API не обращается — используется mock_response, который передаёт сам
узел (planner/generator/critic и т.п.). Мок сознательно живёт рядом с местом,
которое знает, какой ответ разумен для конкретной задачи, а не размазан по
одной универсальной заглушке внутри llm.py.

USE_REAL=True (переменная окружения LLM_USE_REAL) переключает на реальный
вызов через langchain-openai. Провайдер настраивается переменными окружения,
без изменений в коде узлов:
  - OpenAI напрямую:  OPENAI_API_KEY=..., LLM_MODEL=gpt-4o-mini (по умолчанию)
  - OpenRouter:       OPENROUTER_API_KEY=..., LLM_BASE_URL=https://openrouter.ai/api/v1,
                      LLM_MODEL=openai/gpt-4o-mini (или любая другая модель из
                      каталога OpenRouter, формат "провайдер/модель")
OpenRouter API OpenAI-совместим, поэтому ChatOpenAI используется в обоих
случаях — меняется только base_url и ключ.

Единственный контракт, который держим везде: "промпт -> строгий JSON ->
pydantic на стороне вызывающего узла".

Про отказоустойчивость: узлы графа (planner/generator/critic/ranker/
orchestrator) сами решают, откатываться ли на mock_response при LLMError —
но делают это только если разрешено ALLOW_MOCK_FALLBACK (см. ниже). В
реальном режиме (USE_REAL=True) это выключено по умолчанию: сбой должен
быть виден, а не замаскирован правдоподобным мок-контентом.
"""

from __future__ import annotations

import json
import os
import re

USE_REAL = os.getenv("LLM_USE_REAL", "false").strip().lower() in {"1", "true", "yes"}
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# Пусто -> используется дефолтный эндпоинт OpenAI из langchain-openai.
# Заполнено (например, https://openrouter.ai/api/v1) -> запросы идут туда.
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None

# КРИТИЧНО ДЛЯ ПРОДА: разрешён ли узлам тихо откатываться на mock_response,
# если реальный вызов LLM сломался (нет ключа, сеть недоступна, модель
# вернула не-JSON). По умолчанию:
#   - в мок-режиме (USE_REAL=False) это не имеет значения — call_llm_json
#     в этом режиме и так никогда не бросает LLMError, если mock_response передан;
#   - в реальном режиме (USE_REAL=True) по умолчанию FALSE — сбой должен
#     явно ронять прогон, а не тихо подсовывать мок-контент под видом
#     настоящего анализа. Включить откат явно (например, чтобы одна
#     нестабильная гипотеза не рушила весь прогон на публичной демонстрации)
#     можно через LLM_ALLOW_MOCK_FALLBACK=true.
ALLOW_MOCK_FALLBACK = os.getenv(
    "LLM_ALLOW_MOCK_FALLBACK", "false" if USE_REAL else "true"
).strip().lower() in {"1", "true", "yes"}

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(RuntimeError):
    """Ошибка вызова LLM или парсинга её ответа как JSON."""


def _strip_code_fences(text: str) -> str:
    """Убирает markdown-обёртку ```json ... ``` вокруг ответа модели, если она есть."""
    return _CODE_FENCE_RE.sub("", text).strip()


def _resolve_api_key() -> str | None:
    """
    OPENROUTER_API_KEY имеет приоритет, если задан LLM_BASE_URL с OpenRouter —
    иначе используем OPENAI_API_KEY (или ничего, если работаем с локальным
    OpenAI-совместимым эндпоинтом без ключа).
    """
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def _call_real(prompt: str, system: str | None) -> str:
    """Реальный вызов LLM через langchain-openai. Используется только при USE_REAL=True."""
    from langchain_openai import ChatOpenAI  # импорт внутри функции: не требуем пакет/ключ в мок-режиме

    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        base_url=LLM_BASE_URL,
        api_key=_resolve_api_key(),
        temperature=0,
    )
    messages = []
    if system:
        messages.append(("system", system))
    messages.append(("human", prompt))

    try:
        response = llm.invoke(messages)
    except Exception as exc:  # нет ключа, сеть недоступна, лимиты и т.п.
        raise LLMError(f"Реальный вызов LLM не удался: {exc}") from exc

    return response.content


def call_llm_json(
    prompt: str,
    *,
    system: str | None = None,
    mock_response: dict | None = None,
) -> dict:
    """
    Просит LLM вернуть строго JSON и парсит его в dict.

    В мок-режиме (USE_REAL=False, по умолчанию) сразу возвращает
    mock_response — вызывающий узел обязан его передать. В реальном режиме
    вызывает LLM и парсит её ответ как JSON; если ответ не JSON или сам
    вызов не удался, выбрасывает LLMError (узел сам решает, как откатиться
    на мок).
    """
    if not USE_REAL:
        if mock_response is None:
            raise LLMError("USE_REAL=False, но вызывающий узел не передал mock_response")
        return mock_response

    raw = _call_real(prompt, system)
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM вернула не-JSON ответ: {raw!r}") from exc

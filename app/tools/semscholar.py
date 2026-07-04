"""
Внешний источник ссылок — Semantic Scholar.

search_external(query) -> list[Ref] отдаёт список внешних ссылок по теме.
По умолчанию (USE_REAL=False) работает в мок-режиме: без сети и без ключей,
что позволяет прогонять весь граф локально на хакатоне. Чтобы включить
реальный вызов Semantic Scholar Graph API, поставьте переменную окружения
SEMSCHOLAR_USE_REAL=1 (или true/yes) и SEMANTIC_SCHOLAR_KEY в .env — контракт
функции (list[Ref]) не меняется, так что остальной граф не заметит разницы.
"""

from __future__ import annotations

import logging
import os

import requests

from app.state import Ref

logger = logging.getLogger("semscholar")

# Флаг переключения мок/реальный вызов. По умолчанию False — прототип
# должен запускаться без сети и ключей (см. принцип "внешнее — за интерфейсом
# с мок-реализацией").
USE_REAL = os.getenv("SEMSCHOLAR_USE_REAL", "false").strip().lower() in {"1", "true", "yes"}

# Тот же принцип, что и в app/llm.py: в реальном режиме по умолчанию НЕ
# откатываемся тихо на фиктивные [MOCK]-ссылки при сбое сети/ключа — иначе
# в отчёте может незаметно оказаться выдуманная библиография. Включить откат
# явно можно через SEMSCHOLAR_ALLOW_MOCK_FALLBACK=true.
ALLOW_MOCK_FALLBACK = os.getenv(
    "SEMSCHOLAR_ALLOW_MOCK_FALLBACK", "false" if USE_REAL else "true"
).strip().lower() in {"1", "true", "yes"}

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
REQUEST_TIMEOUT_SECONDS = 10


def _mock_search(query: str, limit: int) -> list[Ref]:
    """Детерминированный мок: не бьёт по сети, не требует ключей."""
    return [
        Ref(
            title=f"[MOCK] Публикация по теме «{query}» #{i + 1}",
            url=f"https://example.org/mock-paper/{query.replace(' ', '-')}-{i + 1}",
            year=2024 - i,
        )
        for i in range(limit)
    ]


def _real_search(query: str, limit: int) -> list[Ref]:
    """Реальный запрос к Semantic Scholar Graph API (используется только при USE_REAL=True)."""
    api_key = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}
    params = {"query": query, "limit": limit, "fields": "title,url,year"}

    response = requests.get(
        SEMANTIC_SCHOLAR_API_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    refs: list[Ref] = []
    for paper in payload.get("data", []):
        refs.append(
            Ref(
                title=paper.get("title") or "Без названия",
                url=paper.get("url") or "",
                year=paper.get("year"),
            )
        )
    return refs


def search_external(query: str, limit: int = 5) -> list[Ref]:
    """
    Ищет внешние ссылки по query.

    USE_REAL=False (по умолчанию) -> мок-данные, без сети и ключей.
    USE_REAL=True -> реальный запрос к Semantic Scholar. При сбое по
    умолчанию пробрасывает исключение (ALLOW_MOCK_FALLBACK=False в реальном
    режиме) — недоступность внешнего API не должна незаметно подменяться
    фиктивной библиографией. Откат можно включить явно через
    SEMSCHOLAR_ALLOW_MOCK_FALLBACK=true.
    """
    if not USE_REAL:
        return _mock_search(query, limit)

    try:
        return _real_search(query, limit)
    except Exception as exc:  # сеть недоступна, лимиты, невалидный ключ и т.п.
        if not ALLOW_MOCK_FALLBACK:
            raise
        logger.warning("Semantic Scholar недоступен (%s), использую мок", exc)
        return _mock_search(query, limit)

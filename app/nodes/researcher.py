"""
Узел researcher(state) -> state.

Берёт state["subqueries"] (если планировщик их уже сформировал) или
state["query"] как fallback, вызывает retrieve() по каждому подзапросу и
складывает найденные фрагменты в state["retrieved"], а также
search_external() — во внешние ссылки state["external"].
"""

from __future__ import annotations

from app.retriever import retrieve
from app.state import State
from app.tools.semscholar import search_external


def researcher(state: State) -> State:
    """Наполняет state["retrieved"] и state["external"] по subqueries/query."""
    queries = state["subqueries"] if state["subqueries"] else [state["query"]]

    retrieved = []
    seen_source_ids: set[str] = set()
    external = []

    for q in queries:
        for chunk in retrieve(q):
            if chunk.source_id not in seen_source_ids:
                seen_source_ids.add(chunk.source_id)
                retrieved.append(chunk)
        external.extend(search_external(q))

    state["retrieved"] = retrieved
    state["external"] = external
    state["debate_log"].append(
        f"researcher: retrieved={len(retrieved)} чанков (без дублей по source_id), "
        f"external={len(external)} ссылок, subqueries={len(queries)}"
    )
    return state

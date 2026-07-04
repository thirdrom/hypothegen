"""
Граф LangGraph прототипа: planner, researcher, generator, critic, ranker,
orchestrator — реальные узлы; hitl — проходной HITL-контроль; плюс условный
переход перед ranker.

    planner -> researcher -> generator -> critic
                                  ^          |
                                  | revise   | accept
                                  +----------+---> ranker -> hitl -> orchestrator

planner (app/nodes/planner.py) через app/llm.py (мок по умолчанию) разбивает
state["query"]+constraints на 3-5 подзапросов -> state["subqueries"].
researcher (app/nodes/researcher.py) реально наполняет state["retrieved"]
через retriever.py и state["external"] через app/tools/semscholar.py.
generator (app/nodes/generator.py) через app/llm.py генерирует и валидирует
3 гипотезы (Hypothesis) на основе retrieved/external -> state["hypotheses"],
а на повторных проходах учитывает reasons от critic (адресная доработка).
critic (app/nodes/critic.py) проверяет провенанс (детерминированно, без LLM)
и физико-химическую состоятельность (LLM, мок по умолчанию) каждой гипотезы
-> state["critiques"]; сам же ограничивает число ревизий (MAX_REVISIONS).
ranker (app/nodes/ranker.py + app/ranking.py) оценивает через LLM (мок по
умолчанию) гипотезы с provenance_ok=True и ранжирует их по взвешенному score
-> state["ranked"].
hitl — interrupt() перед orchestrator; ждёт список id одобренных гипотез от
человека (UI/CLI) -> state["approved"].
orchestrator (app/nodes/orchestrator.py) через app/llm.py строит пошаговый
протокол проверки (этапы, ресурсы, критерии успеха/провала) для каждой
одобренной гипотезы -> state["roadmap"].
"""

import sys
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# Позволяет запускать файл напрямую (`python app/graph.py`), а не только
# как модуль пакета (`python -m app.graph`) — добавляем корень проекта в sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.nodes.critic import critic  # noqa: E402
from app.nodes.generator import generator  # noqa: E402
from app.nodes.orchestrator import orchestrator  # noqa: E402
from app.nodes.planner import planner  # noqa: E402
from app.nodes.ranker import ranker  # noqa: E402
from app.nodes.researcher import researcher  # noqa: E402
from app.state import State  # noqa: E402


def route_after_critic(state: State) -> str:
    """
    Условный переход (только чтение state): хотя бы один verdict="revise"
    среди критик -> назад в generator, иначе -> ranker.

    Лимит числа ревизий (MAX_REVISIONS) в это условие сознательно не
    попадает: critic (app/nodes/critic.py) сам гарантирует, что "revise"
    физически не появится среди critiques, как только state["iteration"]
    достигнет MAX_REVISIONS — там же, где происходит инкремент iteration
    (см. подробный комментарий в app/nodes/critic.py про то, почему мутация
    state обязана жить внутри узла, а не в этой routing-функции).
    """
    return "generator" if any(c.verdict == "revise" for c in state["critiques"]) else "ranker"


def hitl_node(state: State) -> State:
    """
    Точка ручного контроля (human-in-the-loop) перед оркестратором.

    interrupt(...) отдаёт человеку (UI/CLI) превью ranked-гипотез и ждёт
    решения. Ожидаемый формат decision — list[str] с id одобренных гипотез;
    он записывается в state["approved"]. Любой другой тип decision (например
    строка "approved" в CLI-демо ниже) просто логируется, approved остаётся
    как было — так интерфейс может присылать частичные/иные сигналы, не
    ломая узел.
    """
    decision = interrupt(
        {
            "message": "Подтвердите одобренные гипотезы (id) перед оркестратором",
            "ranked_preview": [(rh.hypothesis.id, rh.score) for rh in state["ranked"]],
        }
    )
    if isinstance(decision, list):
        state["approved"] = decision
        state["debate_log"].append(f"hitl: пользователь одобрил {len(decision)} гипотез(ы): {decision}")
    else:
        state["debate_log"].append(f"hitl: решение человека = {decision!r}")
    return state


def build_graph():
    """Собирает и компилирует StateGraph с checkpointer'ом (нужен для interrupt)."""
    graph = StateGraph(State)

    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("generator", generator)
    graph.add_node("critic", critic)
    graph.add_node("ranker", ranker)
    graph.add_node("hitl", hitl_node)
    graph.add_node("orchestrator", orchestrator)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "generator")
    graph.add_edge("generator", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"generator": "generator", "ranker": "ranker"},
    )
    graph.add_edge("ranker", "hitl")
    graph.add_edge("hitl", "orchestrator")
    graph.add_edge("orchestrator", END)

    # Явно разрешаем pydantic-модели из app.state в checkpoint-сериализаторе,
    # иначе langgraph выводит deprecation-warning при msgpack-сериализации
    # неизвестных типов (Critique, Chunk, Ref хранятся прямо в state).
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("app.state", "Critique"),
            ("app.state", "Chunk"),
            ("app.state", "Ref"),
            ("app.state", "Hypothesis"),
            ("app.state", "Evidence"),
            ("app.state", "RankedHypothesis"),
            ("app.state", "Roadmap"),
            ("app.state", "RoadmapStep"),
        ]
    )
    return graph.compile(checkpointer=MemorySaver(serde=serde))


if __name__ == "__main__":
    compiled_graph = build_graph()
    config = {"configurable": {"thread_id": "demo-thread"}}

    initial_state: State = {
        "query": "Какие есть гипотезы по снижению себестоимости сплава X на 10%?",
        "constraints": {"budget": "ограничен", "equipment": "существующее"},
        "subqueries": [],
        "retrieved": [],
        "external": [],
        "hypotheses": [],
        "critiques": [],
        "iteration": 0,
        "ranked": [],
        "approved": [],
        "weights": {},
        "debate_log": [],
        "roadmap": [],
    }

    result = compiled_graph.invoke(initial_state, config=config)

    # HITL пока проходной: без реального человека автоматически "одобряем"
    # топ-1 гипотезу по ranked (тот же контракт decision=list[str] id, что
    # использует ui/streamlit_app.py для реального решения человека).
    if "__interrupt__" in result:
        auto_approved = [result["ranked"][0].hypothesis.id] if result["ranked"] else []
        result = compiled_graph.invoke(Command(resume=auto_approved), config=config)

    print("=== debate_log ===")
    for line in result["debate_log"]:
        print(line)

    critic_revise_rounds = sum(
        1
        for line in result["debate_log"]
        if line.startswith("critic:") and "revise=0" not in line
    )
    print(f"\nПрогонов critic с хотя бы одним revise: {critic_revise_rounds}")
    print(f"Финальный iteration: {result['iteration']}")
    print(f"retrieved: {len(result['retrieved'])} чанков")
    print(f"external: {len(result['external'])} ссылок")
    print(f"subqueries: {result['subqueries']}")
    print(f"hypotheses: {len(result['hypotheses'])}")
    print(f"critiques: {[(c.hypothesis_id, c.verdict, c.provenance_ok) for c in result['critiques']]}")
    print(f"ranked: {[(rh.hypothesis.id, rh.score) for rh in result['ranked']]}")
    print(f"approved: {result['approved']}")
    print(f"roadmap: {[(r.hypothesis_id, len(r.steps)) for r in result['roadmap']]}")

"""
Граф сущностей и связей по гипотезам: LLM извлекает тройки
материал -> процесс -> свойство из state["hypotheses"] и state["retrieved"],
дальше это превращается в networkx-граф и рендерится интерактивным pyvis.

Некритичная фича для демо (доп. визуализация в Streamlit), поэтому
реализация сознательно простая: одна pydantic-модель Triple, три функции
(extract_triples, build_graph, render_pyvis_html), без отдельного узла в
LangGraph — вызывается прямо из ui/streamlit_app.py.
"""

from __future__ import annotations

from pydantic import BaseModel
from pyvis.network import Network
import networkx as nx

from app.llm import LLMError, call_llm_json
from app.state import Chunk, Hypothesis

SYSTEM_PROMPT = (
    "Ты — экстрактор сущностей для графа знаний в материаловедении. По "
    "гипотезе и фрагментам источников извлеки тройки (материал, свойство, "
    "процесс): материал — то, с чем работают; свойство — что изменяется "
    "(и есть в kpi_link/статье); процесс — режим/действие, которое связывает "
    "материал со свойством. Если гипотеза не даёт материала/свойства/процесса "
    "явно — верни пустой список, не выдумывай.\n\n"
    "Отвечай СТРОГО в формате JSON без пояснений и markdown-обёртки:\n"
    '{"triples": [{"material": "...", "property": "...", "process": "..."}]}'
)

# Простые маркеры для мок-извлечения (без сети/ключей): ищем в тексте
# гипотезы/evidence слова, обычно обозначающие материал или свойство в
# технических текстах. Это эвристика, а не NLP — как и положено моку.
# Список охватывает и общий пример (сплавы/добавки), и обогащение полезных
# ископаемых (магнетит/футеровка/насадка/хвосты) — при переносе в другой
# домен его нужно будет дополнить под соответствующую терминологию.
_MATERIAL_MARKERS = [
    "сплав", "добавка", "металл", "материал", "карбид", "сырьё", "сырье",
    "магнетит", "руда", "порода", "футеровка", "насадка", "концентрат",
]
_PROPERTY_MARKERS = [
    "жаропрочность",
    "прочность",
    "себестоимость",
    "стоимость",
    "пластичность",
    "коррозионная стойкость",
    "выход",
    "потери",
    "извлечение",
    "содержание",
]


class Triple(BaseModel):
    """Одна связь материал-свойство-процесс, извлечённая из одной гипотезы."""

    material: str
    property: str
    process: str
    hypothesis_id: str


def _find_marker(text: str, markers: list[str]) -> str | None:
    text_lower = text.lower()
    for marker in markers:
        if marker in text_lower:
            return marker
    return None


def _mock_extract(hypothesis: Hypothesis, chunk_by_source_id: dict[str, Chunk]) -> dict:
    """
    Детерминированный мок без сети/ключей: ищет материал- и свойство-маркеры
    в statement, evidence.fact И сыром тексте исходных чанков (retrieved) —
    пересказ в evidence.fact мог не сохранить нужное слово, а в оригинале
    источника оно есть. Процесс берёт из conditions/derivation_method.
    Если не нашлось и материала, и свойства — возвращает пустой список
    (честно: не на чем строить тройку, не выдумываем сущности из воздуха).
    """
    raw_chunks_text = " ".join(
        chunk_by_source_id[e.source_id].text for e in hypothesis.evidence if e.source_id in chunk_by_source_id
    )
    haystack = hypothesis.statement + " " + " ".join(e.fact for e in hypothesis.evidence) + " " + raw_chunks_text

    material = _find_marker(haystack, _MATERIAL_MARKERS)
    property_ = _find_marker(haystack, _PROPERTY_MARKERS)

    if not material or not property_:
        return {"triples": []}

    if hypothesis.conditions:
        process = ", ".join(f"{k}={v}" for k, v in hypothesis.conditions.items())
    else:
        process = hypothesis.derivation_method

    return {
        "triples": [
            {"material": material, "property": property_, "process": process},
        ]
    }


def _build_prompt(hypothesis: Hypothesis, chunk_by_source_id: dict[str, Chunk]) -> str:
    facts = "\n".join(f"- {e.fact}" for e in hypothesis.evidence)
    source_excerpts = "\n".join(
        f"- [{e.source_id}] {chunk_by_source_id[e.source_id].text[:300]}"
        for e in hypothesis.evidence
        if e.source_id in chunk_by_source_id
    )
    return (
        f"Гипотеза: {hypothesis.statement}\n"
        f"Факты (пересказ): {facts}\n"
        f"Оригинальный текст источников:\n{source_excerpts or '(не найден)'}\n"
        f"Условия (conditions): {hypothesis.conditions}\n"
        f"KPI: {hypothesis.kpi_link}"
    )


def extract_triples(hypotheses: list[Hypothesis], retrieved: list[Chunk]) -> list[Triple]:
    """
    Извлекает тройки материал-свойство-процесс из гипотез и их источников
    через LLM (мок по умолчанию). Для каждой гипотезы подтягивает сырой
    текст чанков (retrieved), на которые ссылается её evidence, — и мок, и
    реальный промпт видят не только пересказ (evidence.fact), но и оригинал.
    """
    chunk_by_source_id = {c.source_id: c for c in retrieved}

    triples: list[Triple] = []
    for hypothesis in hypotheses:
        mock_response = _mock_extract(hypothesis, chunk_by_source_id)
        prompt = _build_prompt(hypothesis, chunk_by_source_id)
        try:
            response = call_llm_json(prompt, system=SYSTEM_PROMPT, mock_response=mock_response)
        except LLMError:
            response = mock_response

        if not isinstance(response, dict):
            response = mock_response

        for raw in response.get("triples") or []:
            try:
                triples.append(
                    Triple(
                        material=str(raw["material"]),
                        property=str(raw["property"]),
                        process=str(raw["process"]),
                        hypothesis_id=hypothesis.id,
                    )
                )
            except Exception:
                continue  # некорректная тройка от LLM — пропускаем, не рушим остальные

    return triples


def build_graph(triples: list[Triple]) -> nx.DiGraph:
    """
    Строит граф material -> process -> property. Узлы помечены атрибутом
    "kind" (material/process/property) для раскраски в pyvis. Повторяющиеся
    рёбра не дублируются — вместо этого копится список hypothesis_id.
    """
    graph = nx.DiGraph()

    for triple in triples:
        graph.add_node(triple.material, kind="material")
        graph.add_node(triple.process, kind="process")
        graph.add_node(triple.property, kind="property")

        for u, v, relation in (
            (triple.material, triple.process, "участвует в"),
            (triple.process, triple.property, "влияет на"),
        ):
            if graph.has_edge(u, v):
                graph[u][v]["hypotheses"].append(triple.hypothesis_id)
            else:
                graph.add_edge(u, v, relation=relation, hypotheses=[triple.hypothesis_id])

    return graph


_KIND_COLORS = {
    "material": "#4C72B0",
    "process": "#DD8452",
    "property": "#55A868",
}


def render_pyvis_html(graph: nx.DiGraph) -> str:
    """Рендерит граф в самодостаточный HTML (JS/CSS встроены) для st.components.v1.html."""
    net = Network(height="600px", width="100%", directed=True, cdn_resources="in_line")

    for node, data in graph.nodes(data=True):
        kind = data.get("kind", "?")
        net.add_node(node, label=node, color=_KIND_COLORS.get(kind, "#888888"), title=f"{kind}: {node}")

    for u, v, data in graph.edges(data=True):
        hyps = data.get("hypotheses", [])
        title = f"{data.get('relation', '')} (гипотезы: {', '.join(hyps)})"
        net.add_edge(u, v, label=data.get("relation", ""), title=title)

    net.repulsion(node_distance=180, spring_length=200)
    return net.generate_html(notebook=False)

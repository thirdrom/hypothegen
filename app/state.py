"""
Контракт данных всего проекта. Единственный источник правды: все узлы графа
(app/nodes/*) и app/graph.py импортируют модели и State только отсюда.

Pydantic-модели описывают "предметные" сущности (чанк корпуса, внешняя ссылка,
свидетельство, гипотеза с полной трассировкой вывода, критика, ранжированная
гипотеза). TypedDict State — это состояние графа LangGraph, которое передаётся
между узлами.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Один фрагмент корпуса, полученный на этапе ingest и найденный retriever'ом."""

    text: str
    source: str
    source_id: str
    metadata: dict = Field(default_factory=dict)


class Ref(BaseModel):
    """Внешняя ссылка (например, из Semantic Scholar)."""

    title: str
    url: str
    year: int | None = None


class Evidence(BaseModel):
    """
    Одно свидетельство в пользу гипотезы.

    source_id — ссылка на конкретный retrieved-фрагмент (Chunk.source_id);
    fact — пересказ факта из этого фрагмента своими словами;
    how_used — как именно этот факт привёл к формулировке гипотезы.
    """

    source_id: str
    fact: str
    how_used: str


class Hypothesis(BaseModel):
    """
    Гипотеза с полной трассировкой вывода: как сформулирована, почему именно
    она, при каких условиях работает, и итоговые оценки от Критика/Ранкера.
    """

    id: str
    statement: str

    # --- КАК сформулировано (провенанс / трассировка вывода) ---
    derivation_method: Literal[
        "analogy", "knowledge_gap", "counterfactual", "extrapolation", "combination"
    ]
    evidence: list[Evidence] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)

    # --- ПОЧЕМУ именно это ---
    kpi_link: str
    novelty_justification: str
    rejected_alternatives: list[str] = Field(default_factory=list)

    # --- ПРИ КАКИХ УСЛОВИЯХ работать ---
    conditions: dict = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    constraints_satisfied: dict = Field(default_factory=dict)
    validity_limits: str

    # --- оценки (заполняют Критик и Ранкер) ---
    novelty: float = 0.0
    value: float = 0.0
    feasibility: float = 0.0
    risk: float = 0.0
    cost_of_error: float = 0.0
    rationale: str = ""


class Critique(BaseModel):
    """Вердикт Критика по одной гипотезе."""

    hypothesis_id: str
    verdict: Literal["accept", "revise", "reject"]
    reasons: list[str] = Field(default_factory=list)
    flagged_risks: list[str] = Field(default_factory=list)
    provenance_ok: bool  # прошла ли гипотеза проверку трассируемости (evidence на каждый шаг)


class RankedHypothesis(BaseModel):
    """Гипотеза со скорингом от Ранкера."""

    hypothesis: Hypothesis
    score: float


class RoadmapStep(BaseModel):
    """Один этап протокола проверки одобренной гипотезы."""

    name: str
    resources: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    failure_criteria: list[str] = Field(default_factory=list)


class Roadmap(BaseModel):
    """Пошаговый протокол проверки одной одобренной гипотезы (строит Оркестратор)."""

    hypothesis_id: str
    steps: list[RoadmapStep] = Field(default_factory=list)


class State(TypedDict):
    """Состояние графа LangGraph, передаваемое между узлами."""

    query: str
    constraints: dict
    subqueries: list[str]
    retrieved: list[Chunk]
    external: list[Ref]
    hypotheses: list[Hypothesis]
    critiques: list[Critique]
    iteration: int
    ranked: list[RankedHypothesis]
    approved: list[str]
    weights: dict
    debate_log: list[str]
    roadmap: list[Roadmap]

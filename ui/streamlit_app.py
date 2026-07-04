"""
Streamlit UI прототипа.

Форма ввода (query + constraints) запускает граф (app/graph.py) до
интеррапта перед orchestrator (узел hitl). Пока граф стоит на интеррапте,
показываем:
  1. state["debate_log"] — лог дебатов узлов;
  2. слайдеры весов формулы ранжирования — при изменении локально
     пересчитываем rank() (без LLM и без повторного прогона графа) и
     показываем новый порядок сразу;
  3. ранжированный список гипотез с обоснованием (rationale, provenance) и
     источниками (evidence -> source_id/source из state["retrieved"]);
  4. чекбокс "одобрить" по каждой гипотезе (approve/block).

По кнопке "Подтвердить и продолжить" резюмируем граф через
Command(resume=<id одобренных гипотез>) — hitl-узел кладёт этот список в
state["approved"], граф доходит до orchestrator и завершается.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

# Позволяет запускать `streamlit run ui/streamlit_app.py` из любой директории:
# добавляем корень проекта в sys.path, иначе `from app...` не найдётся.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ВАЖНО: Streamlit Cloud хранит ключи в st.secrets (TOML-форма в веб-интерфейсе),
# а НЕ в переменных окружения. Но app/llm.py и app/tools/semscholar.py читают
# конфигурацию через os.getenv(...) на уровне модуля — то есть один раз, в
# момент импорта. Поэтому секреты нужно скопировать в os.environ ДО того, как
# ниже импортируется app.graph (который импортирует app.llm). Локально, если
# .streamlit/secrets.toml не создан, st.secrets кидает исключение при попытке
# перечислить ключи — это ожидаемо (обычный локальный запуск использует
# переменные окружения/.env, а не Streamlit secrets), просто пропускаем.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from app.entity_graph import build_graph as build_entity_graph  # noqa: E402
from app.entity_graph import extract_triples, render_pyvis_html  # noqa: E402
from app.export import to_docx, to_pdf, to_tasks_csv, to_tasks_json  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.ingest import SUPPORTED_EXTENSIONS, ingest  # noqa: E402
from app.ranking import DEFAULT_WEIGHTS, rank  # noqa: E402
from langgraph.types import Command  # noqa: E402

# Та же папка, что и для `python -m app.ingest data` из README — единственная
# точка правды для корпуса что при запуске из терминала, что при загрузке
# файлов через браузер ниже. Файлы, загруженные через UI, физически кладутся
# сюда же и переиндексируются вместе с тем, что уже лежит в data/ на диске.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(page_title="Генератор гипотез", layout="wide")


@st.cache_resource
def get_compiled_graph():
    """Граф компилируется один раз на процесс Streamlit (MemorySaver общий на все сессии,
    но у каждой сессии свой thread_id, так что треды не пересекаются)."""
    return build_graph()


def parse_constraints(text: str) -> dict:
    """Строки вида "ключ: значение" -> dict. Пустые строки и строки без ':' игнорируются."""
    constraints = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key:
            constraints[key] = value
    return constraints


def reset_session() -> None:
    """Полный сброс: новый thread_id, новая пустая сессия, назад к форме ввода."""
    st.session_state.stage = "input"
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pop("interrupted_state", None)
    st.session_state.pop("final_state", None)


if "stage" not in st.session_state:
    reset_session()

graph = get_compiled_graph()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("Генератор исследовательских гипотез")

# ---------------------------------------------------------------------------
# 1. Форма ввода
# ---------------------------------------------------------------------------
if st.session_state.stage == "input":
    with st.expander("📁 База знаний"):
        existing_files = sorted(
            p.name for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ) if DATA_DIR.exists() else []
        if existing_files:
            st.caption(f"Сейчас в data/ уже лежит {len(existing_files)} файл(ов): {', '.join(existing_files)}")
        else:
            st.caption("data/ сейчас пуста — без файлов поиск по источникам не найдёт ничего (retrieved=0).")

        uploaded_files = st.file_uploader(
            "Загрузить документы (PDF/XLSX/XLS/CSV/TXT) — добавятся к тому, что уже есть в data/",
            type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Загрузить и переиндексировать"):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for uploaded in uploaded_files:
                (DATA_DIR / uploaded.name).write_bytes(uploaded.getvalue())
            with st.spinner("Индексирую data/ (полная пересборка, файлы с тем же именем перезаписываются)..."):
                ingest(str(DATA_DIR))
            st.success(f"Загружено {len(uploaded_files)} файл(ов) и переиндексировано.")
            st.rerun()

    with st.form("query_form"):
        query = st.text_area(
            "Задача",
            placeholder="Как снизить себестоимость сплава X на 10% без потери жаропрочности?",
            height=100,
        )
        constraints_text = st.text_area(
            "Ограничения (по одному на строку, формат «ключ: значение»)",
            placeholder="budget: ограничен\nequipment: существующее",
            height=80,
        )
        submitted = st.form_submit_button("Запустить граф")

    if submitted:
        if not query.strip():
            st.warning("Введите задачу — поле не может быть пустым.")
        else:
            initial_state = {
                "query": query.strip(),
                "constraints": parse_constraints(constraints_text),
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
            with st.spinner("Граф думает: planner -> researcher -> generator -> critic -> ranker..."):
                result = graph.invoke(initial_state, config=config)

            st.session_state.interrupted_state = result
            st.session_state.stage = "interrupted" if "__interrupt__" in result else "done"
            if st.session_state.stage == "done":
                st.session_state.final_state = result
            st.rerun()

# ---------------------------------------------------------------------------
# 2-4. Граф остановлен перед orchestrator: дебаты, веса, ранжирование, approve/block
# ---------------------------------------------------------------------------
elif st.session_state.stage == "interrupted":
    state = st.session_state.interrupted_state

    with st.expander("Лог дебатов (debate_log)", expanded=False):
        for line in state["debate_log"]:
            st.text(line)

    eligible_hypotheses = [rh.hypothesis for rh in state["ranked"]]
    source_lookup = {c.source_id: c.source for c in state["retrieved"]}

    st.subheader("Веса формулы ранжирования")
    st.caption("score = wn·novelty + wv·value + wf·feasibility − wr·risk − we·cost_of_error")
    weight_cols = st.columns(5)
    weight_labels = {
        "novelty": "Новизна (wn)",
        "value": "Ценность (wv)",
        "feasibility": "Реализуемость (wf)",
        "risk": "Риск (wr)",
        "cost_of_error": "Цена ошибки (we)",
    }
    weights = {}
    for col, (key, label) in zip(weight_cols, weight_labels.items()):
        with col:
            weights[key] = st.slider(label, 0.0, 1.0, DEFAULT_WEIGHTS[key], 0.05, key=f"weight_{key}")

    if not eligible_hypotheses:
        st.info("Нет гипотез с валидным провенансом для ранжирования (все отклонены критиком).")
        ranked_now = []
    else:
        # Переранжирование — чисто локальный пересчёт по уже выставленным
        # компонентам (novelty/value/...), без LLM и без повторного прогона
        # графа. Срабатывает на каждое движение слайдера (Streamlit
        # перезапускает скрипт на любое изменение виджета).
        ranked_now = rank(eligible_hypotheses, weights=weights)

    st.subheader(f"Ранжированные гипотезы ({len(ranked_now)})")

    approvals: dict[str, bool] = {}
    for position, ranked_hypothesis in enumerate(ranked_now, start=1):
        h = ranked_hypothesis.hypothesis
        with st.container(border=True):
            header_col, checkbox_col = st.columns([5, 1])
            with header_col:
                st.markdown(f"**#{position} · {h.id} · score={ranked_hypothesis.score:.4f}**")
                st.write(h.statement)
            with checkbox_col:
                approvals[h.id] = st.checkbox("Одобрить", key=f"approve_{h.id}")

            st.caption(
                f"novelty={h.novelty} · value={h.value} · feasibility={h.feasibility} "
                f"· risk={h.risk} · cost_of_error={h.cost_of_error}"
            )
            st.markdown(f"_Обоснование оценки:_ {h.rationale}")

            with st.expander("Провенанс и условия"):
                st.markdown("**Evidence (источники):**")
                for e in h.evidence:
                    human_source = source_lookup.get(e.source_id, "источник не найден в retrieved")
                    st.markdown(f"- `{e.source_id}` ({human_source}): {e.fact} — _{e.how_used}_")
                st.markdown("**Цепочка рассуждений:**")
                for i, step in enumerate(h.reasoning_steps, start=1):
                    st.markdown(f"{i}. {step}")
                st.markdown(f"**KPI:** {h.kpi_link}")
                st.markdown(f"**Почему не альтернативы:** {'; '.join(h.rejected_alternatives)}")
                st.markdown(f"**Условия:** {h.conditions}")
                st.markdown(f"**Предположения:** {'; '.join(h.assumptions) or '—'}")
                st.markdown(f"**Пределы применимости:** {h.validity_limits}")

    st.subheader("Граф сущностей (материал → процесс → свойство)")
    st.caption("Некритичная визуализация: LLM-извлечение пар материал–свойство–процесс из гипотез и источников.")
    hyp_ids_key = tuple(h.id for h in eligible_hypotheses)
    if st.session_state.get("entity_graph_key") != hyp_ids_key:
        triples = extract_triples(eligible_hypotheses, state["retrieved"])
        st.session_state.entity_graph_key = hyp_ids_key
        st.session_state.entity_graph_triples = len(triples)
        st.session_state.entity_graph_html = (
            render_pyvis_html(build_entity_graph(triples)) if triples else None
        )

    if st.session_state.entity_graph_html:
        st.iframe(st.session_state.entity_graph_html, height=620)
    else:
        st.info("Не удалось извлечь сущности материал/свойство из текущих гипотез.")

    st.divider()
    if st.button("Подтвердить и продолжить", type="primary"):
        approved_ids = [hyp_id for hyp_id, is_approved in approvals.items() if is_approved]
        with st.spinner("Передаю решение оркестратору..."):
            final_result = graph.invoke(Command(resume=approved_ids), config=config)
        st.session_state.final_state = final_result
        st.session_state.stage = "done"
        st.rerun()

    if st.button("Начать заново"):
        reset_session()
        st.rerun()

# ---------------------------------------------------------------------------
# Граф завершён (дошёл до orchestrator)
# ---------------------------------------------------------------------------
elif st.session_state.stage == "done":
    state = st.session_state.final_state

    st.success(f"Граф завершён. Одобрено гипотез: {len(state['approved'])} — {state['approved']}")

    st.subheader("Экспорт")
    export_dir = Path(tempfile.gettempdir()) / f"hypothesis-export-{st.session_state.thread_id}"
    export_dir.mkdir(parents=True, exist_ok=True)
    docx_path = export_dir / "report.docx"
    pdf_path = export_dir / "report.pdf"
    csv_path = export_dir / "tasks.csv"
    json_path = export_dir / "tasks.json"

    with st.spinner("Собираю отчёты..."):
        to_docx(state, str(docx_path))
        to_pdf(state, str(pdf_path))
        to_tasks_csv(state, str(csv_path))
        to_tasks_json(state, str(json_path))

    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
    with export_col1:
        st.download_button(
            "📄 Отчёт DOCX",
            data=docx_path.read_bytes(),
            file_name="report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    with export_col2:
        st.download_button(
            "📄 Отчёт PDF",
            data=pdf_path.read_bytes(),
            file_name="report.pdf",
            mime="application/pdf",
        )
    with export_col3:
        st.download_button(
            "🗂️ Задачи CSV",
            data=csv_path.read_bytes(),
            file_name="tasks.csv",
            mime="text/csv",
        )
    with export_col4:
        st.download_button(
            "🗂️ Задачи JSON",
            data=json_path.read_bytes(),
            file_name="tasks.json",
            mime="application/json",
        )

    with st.expander("Полный лог дебатов (debate_log)", expanded=True):
        for line in state["debate_log"]:
            st.text(line)

    if state["ranked"]:
        st.subheader("Финальный рейтинг")
        for position, rh in enumerate(state["ranked"], start=1):
            mark = "✅" if rh.hypothesis.id in state["approved"] else "—"
            st.markdown(f"{mark} **#{position} · {rh.hypothesis.id}** · score={rh.score:.4f}: {rh.hypothesis.statement}")

    if state["roadmap"]:
        st.subheader("Протоколы проверки одобренных гипотез")
        for roadmap in state["roadmap"]:
            with st.expander(f"Протокол для {roadmap.hypothesis_id} ({len(roadmap.steps)} этапов)"):
                for i, step in enumerate(roadmap.steps, start=1):
                    st.markdown(f"**{i}. {step.name}**")
                    st.markdown(f"- Ресурсы: {'; '.join(step.resources) or '—'}")
                    st.markdown(f"- Критерии успеха: {'; '.join(step.success_criteria) or '—'}")
                    st.markdown(f"- Критерии провала: {'; '.join(step.failure_criteria) or '—'}")

    if st.button("Начать заново"):
        reset_session()
        st.rerun()

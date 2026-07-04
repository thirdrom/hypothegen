# Прототип: генератор и трассируемая проверка исследовательских гипотез

Хакатон, ~24ч. Стек: Python 3.11, LangGraph, LangChain, ChromaDB, Streamlit,
PyMuPDF, pandas, python-docx, reportlab, requests, pydantic, networkx, pyvis.

## Принципы проекта

- Один ответ = один файл или одна функция. Без лишних абстракций и "агентов".
- Всё внешнее (LLM, Semantic Scholar) спрятано за интерфейс с мок-реализацией —
  код запускается без ключей и сети (`app/llm.py`, `app/tools/semscholar.py`).
- LLM всегда просим вернуть строгий JSON, парсим в pydantic-модели (`app/state.py`).
- В реальном режиме (`LLM_USE_REAL=true`) сбой LLM/внешнего API **явно роняет
  прогон**, а не тихо подменяется моком — см. раздел "Мок vs реальный режим".
- Никаких TODO-заглушек в местах, реализованных по явному запросу.

## Структура проекта

```
app/
  state.py         # ЕДИНЫЙ источник правды: Chunk, Ref, Evidence, Hypothesis,
                    # Critique, RankedHypothesis, Roadmap(Step), State (TypedDict)
  llm.py           # единая точка вызова LLM: мок по умолчанию, OpenAI/OpenRouter
                    # за флагом LLM_USE_REAL
  ingest.py        # PDF/Excel/CSV/TXT -> чанки -> ChromaDB (./chroma)
  retriever.py     # retrieve(query, k) -> list[Chunk] из ChromaDB
  ranking.py       # score_hypothesis() (LLM-оценщик) + rank() (взвешенный score)
  export.py        # to_docx / to_pdf / to_tasks_csv / to_tasks_json
  entity_graph.py  # LLM-извлечение троек материал-свойство-процесс + networkx/pyvis
  graph.py         # сборка графа LangGraph, CLI-демо (python app/graph.py)
  tools/
    semscholar.py  # search_external(): мок по умолчанию, реальный API за флагом
  nodes/           # один узел — один файл
    planner.py       # query+constraints -> 3-5 subqueries
    researcher.py    # subqueries -> retrieved + external
    generator.py     # retrieved+external -> N=3 валидные Hypothesis
    critic.py        # провенанс (без LLM) + LLM-ревью -> Critique, лимит ревизий
    ranker.py        # фильтр provenance_ok + score_hypothesis + rank -> ranked
    orchestrator.py  # approved -> Roadmap (протокол проверки) через LLM
  assets/fonts/    # DejaVu Sans — кириллица в PDF-отчётах (reportlab)
ui/
  streamlit_app.py # форма ввода -> граф -> дебаты/веса/ранжирование -> HITL -> экспорт
data/              # пустая папка под корпус документов (.pdf/.xlsx/.csv/.txt)
requirements.txt
.env.example
README.md
```

## Граф

```
planner -> researcher -> generator -> critic
                              ^          |
                              | revise   | accept
                              +----------+---> ranker -> hitl(interrupt) -> orchestrator
```

## Установка

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Мок vs реальный режим

По умолчанию **всё работает без ключей и сети** (`.env` можно не трогать).
Чтобы включить настоящий LLM:

```bash
# Вариант А: OpenAI напрямую
export LLM_USE_REAL=true
export OPENAI_API_KEY=sk-...

# Вариант Б: OpenRouter (любая модель каталога, формат "провайдер/модель")
export LLM_USE_REAL=true
export OPENROUTER_API_KEY=sk-or-...
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_MODEL=openai/gpt-4o-mini
```

В реальном режиме сбой (нет ключа, сеть недоступна, не-JSON ответ) **явно
роняет прогон** с полным traceback — по умолчанию узлы не откатываются на
мок молча. Если для публичной демонстрации нужна устойчивость ценой
прозрачности — включите явно: `LLM_ALLOW_MOCK_FALLBACK=true` (и/или
`SEMSCHOLAR_ALLOW_MOCK_FALLBACK=true`).

## Как запустить

### 1. Положить корпус и проиндексировать

```bash
# положите 2+ файлов (.pdf/.xlsx/.csv/.txt) в data/, затем:
python -m app.ingest data
```
Ожидаемый вывод: `INFO: <файл>: проиндексировано N чанков` по каждому файлу,
в конце `Проиндексировано чанков: <итого>`.

### 2. CLI-демо (весь граф целиком, без UI)

```bash
python app/graph.py
```
Ожидаемый вывод (хвост): лог по каждому узлу в `debate_log`, затем сводка —
```
Прогонов critic с хотя бы одним revise: 2
Финальный iteration: 2
retrieved: N чанков
external: N ссылок
subqueries: [...]
hypotheses: 3
critiques: [(id, verdict, provenance_ok), ...]
ranked: [(id, score), ...]
approved: [...]
roadmap: [(id, число_шагов), ...]
```

### 3. Streamlit UI (интерактивно, с HITL и экспортом)

```bash
streamlit run ui/streamlit_app.py
```
Откроется форма: задача + ограничения -> «Запустить граф» -> дебаты/слайдеры
весов/граф сущностей/список гипотез с чекбоксами -> «Подтвердить и
продолжить» -> финальный рейтинг + 4 кнопки скачивания (DOCX/PDF/CSV/JSON).

## Как тестировать по частям

Каждый модуль можно проверить изолированно, не поднимая весь граф:

```bash
# retriever (после ingest)
python -c "from app.retriever import retrieve; print(retrieve('ваш запрос'))"

# планировщик
python -c "
from app.nodes.planner import planner
s = {'query': 'тест', 'constraints': {}, 'subqueries': [], 'retrieved': [], 'external': [],
     'hypotheses': [], 'critiques': [], 'iteration': 0, 'ranked': [], 'approved': [],
     'weights': {}, 'debate_log': [], 'roadmap': []}
print(planner(s)['subqueries'])
"

# критик и ранкер — см. app/nodes/critic.py и app/ranking.py, там есть
# примеры прямого вызова в докстроках/комментариях

# экспорт (нужен непустой state после полного прогона графа)
python -c "
from app.graph import build_graph
from langgraph.types import Command
g = build_graph()
cfg = {'configurable': {'thread_id': 't'}}
r = g.invoke({'query': 'тест', 'constraints': {}, 'subqueries': [], 'retrieved': [],
              'external': [], 'hypotheses': [], 'critiques': [], 'iteration': 0,
              'ranked': [], 'approved': [], 'weights': {}, 'debate_log': [], 'roadmap': []}, config=cfg)
r = g.invoke(Command(resume=[r['ranked'][0].hypothesis.id] if r['ranked'] else []), config=cfg)
from app.export import to_docx, to_pdf, to_tasks_csv, to_tasks_json
to_docx(r, '/tmp/report.docx'); to_pdf(r, '/tmp/report.pdf')
to_tasks_csv(r, '/tmp/tasks.csv'); to_tasks_json(r, '/tmp/tasks.json')
print('готово')
"
```

### Streamlit без браузера (для CI / быстрой проверки после правок)

```bash
python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('ui/streamlit_app.py')
at.run(timeout=30)
print('exceptions:', at.exception)
"
```
Ожидаемый вывод: `exceptions: ElementList()` (пусто — значит, страница
отрисовалась без ошибок).

## Критерий готовности установки

```bash
pip install -r requirements.txt
```
должен пройти без ошибок — это единственное жёсткое условие для проверки
структуры проекта; остальное проверяется запуском выше.

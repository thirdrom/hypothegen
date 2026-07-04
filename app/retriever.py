"""
Поиск по локальному ChromaDB (./chroma, коллекция "corpus", см. app/ingest.py).

retrieve(query, k=5) возвращает k ближайших чанков как объекты Chunk из
app/state.py — с заполненными source и source_id, восстановленными из
метаданных ChromaDB. source_id обязателен: на него ссылается Evidence при
формулировке гипотез (app/state.py -> Evidence.source_id).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

# Позволяет запускать файл напрямую (`python app/retriever.py`), а не только
# как модуль пакета (`python -m app.retriever`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import CHROMA_PERSIST_DIR, COLLECTION_NAME, HashingEmbedding  # noqa: E402
from app.state import Chunk  # noqa: E402

logger = logging.getLogger("retriever")


def retrieve(query: str, k: int = 5) -> list[Chunk]:
    """
    Ищет k ближайших чанков в локальном ChromaDB по query.

    Использует ту же HashingEmbedding, что и ingest.py: коллекция была
    проиндексирована с этой embedding-функцией, и запрос обязан кодироваться
    той же функцией, иначе векторы будут несовместимы.

    Если коллекция ещё не создана (ingest не запускался) или пуста,
    возвращает пустой список, а не падает — это ожидаемый, штатный случай.

    ВАЖНО: любая ДРУГАЯ ошибка (повреждённая база, рассинхрон версии
    chromadb между запуском ingest и retrieve, проблемы с правами на
    ./chroma и т.п.) намеренно НЕ проглатывается — раньше здесь был широкий
    `except Exception: return []`, из-за которого любая реальная поломка
    выглядела как "источников просто нет", и это было невозможно
    диагностировать. Теперь такая ошибка логируется и пробрасывается дальше.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=HashingEmbedding())
    except NotFoundError:
        logger.info("Коллекция '%s' ещё не создана — ingest не запускался или база пуста", COLLECTION_NAME)
        return []
    except Exception:
        logger.exception(
            "Не удалось открыть коллекцию '%s' в %s — это НЕ штатный случай "
            "'источников нет', а реальная ошибка (см. traceback выше)",
            COLLECTION_NAME,
            CHROMA_PERSIST_DIR,
        )
        raise

    count = collection.count()
    if count == 0:
        return []

    n_results = min(k, count)
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
    except Exception:
        logger.exception("collection.query() упал для query=%r", query)
        raise

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    chunks: list[Chunk] = []
    for text, metadata in zip(documents, metadatas):
        metadata = dict(metadata)
        source = metadata.pop("source", "")
        source_id = metadata.pop("source_id", "")
        chunks.append(Chunk(text=text, source=source, source_id=source_id, metadata=metadata))
    return chunks


if __name__ == "__main__":
    query_arg = sys.argv[1] if len(sys.argv) > 1 else "жаропрочность"
    found = retrieve(query_arg)
    print(f"Найдено чанков: {len(found)}")
    for chunk in found:
        preview = chunk.text[:80].replace("\n", " ")
        print(f"[{chunk.source_id}] {chunk.source}: {preview!r}")
        
"""
Поиск по локальному ChromaDB (./chroma, коллекция "corpus", см. app/ingest.py).

retrieve(query, k=5) возвращает k ближайших чанков как объекты Chunk из
app/state.py — с заполненными source и source_id, восстановленными из
метаданных ChromaDB. source_id обязателен: на него ссылается Evidence при
формулировке гипотез (app/state.py -> Evidence.source_id).
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb

# Позволяет запускать файл напрямую (`python app/retriever.py`), а не только
# как модуль пакета (`python -m app.retriever`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import CHROMA_PERSIST_DIR, COLLECTION_NAME, HashingEmbedding  # noqa: E402
from app.state import Chunk  # noqa: E402


def retrieve(query: str, k: int = 5) -> list[Chunk]:
    """
    Ищет k ближайших чанков в локальном ChromaDB по query.

    Использует ту же HashingEmbedding, что и ingest.py: коллекция была
    проиндексирована с этой embedding-функцией, и запрос обязан кодироваться
    той же функцией, иначе векторы будут несовместимы.

    Если коллекция ещё не создана (ingest не запускался) или пуста,
    возвращает пустой список, а не падает.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=HashingEmbedding())
    except Exception:
        return []

    count = collection.count()
    if count == 0:
        return []

    n_results = min(k, count)
    results = collection.query(query_texts=[query], n_results=n_results)

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

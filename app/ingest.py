"""
Загрузка корпуса (PDF/Excel/CSV/TXT) в локальный ChromaDB.

Функция ingest(folder) читает все поддерживаемые файлы из папки, режет их на
чанки ~800 символов с перекрытием ~100 символов и сохраняет их в персистентную
коллекцию ChromaDB (./chroma). Каждому чанку присваивается стабильный
source_id вида "<имя_файла>#<n>" (n — сквозной номер чанка внутри файла) —
по нему позже строятся Evidence в гипотезах (см. app/state.py). В метаданные
чанка также пишется человекочитаемый source (файл + страница/лист).

Про эмбеддинги: дефолтная embedding-функция ChromaDB (ONNX MiniLM) при первом
использовании скачивает модель с интернета — это ломает принцип "код должен
запускаться без ключей и сети". Поэтому здесь используется собственная
детерминированная офлайн-embedding-функция HashingEmbedding (без сети и
внешних моделей). Она прячется за тем же интерфейсом chromadb.EmbeddingFunction,
так что позже её можно заменить на реальную (OpenAI/sentence-transformers),
не меняя ingest/retriever логику — тот же принцип "внешнее за интерфейсом",
что и для LLM.

Битые/пустые файлы логируются и пропускаются — процесс не падает.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import chromadb
import fitz  # PyMuPDF
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

# Позволяет запускать файл напрямую (`python app/ingest.py`), а не только
# как модуль пакета (`python -m app.ingest`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.state import Chunk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "chroma")
COLLECTION_NAME = "corpus"
EMBEDDING_DIM = 256

SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv", ".txt"}


@register_embedding_function
class HashingEmbedding(EmbeddingFunction[Documents]):
    """
    Детерминированная офлайн-embedding-функция без сети и внешних моделей.

    Хэширует слова текста в EMBEDDING_DIM корзин (bag-of-words hashing trick)
    и нормализует вектор. Она не такая качественная, как настоящая
    embedding-модель, но полностью офлайн, стабильна между запусками и
    достаточна, чтобы прототип на хакатоне работал без ключей и интернета.
    Реальную embedding-модель можно подключить позже за тем же интерфейсом.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "hashing_embedding_v1"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashingEmbedding":
        return HashingEmbedding()

    def get_config(self) -> dict[str, Any]:
        return {}

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 (имя параметра задано интерфейсом chromadb)
        return [self._embed_one(text) for text in input]

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        words = text.lower().split()
        if not words:
            return vector
        for word in words:
            bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM
            vector[bucket] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Режет текст на перекрывающиеся чанки по chunk_size символов."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def _read_pdf(path: Path) -> list[tuple[str, str]]:
    """Возвращает список (текст_страницы, человекочитаемый_source) по PDF."""
    sections: list[tuple[str, str]] = []
    try:
        doc = fitz.open(path)
    except Exception as exc:
        logger.warning("Пропускаю %s: не удалось открыть PDF (%s)", path.name, exc)
        return sections

    try:
        for page_number, page in enumerate(doc, start=1):
            try:
                text = page.get_text()
            except Exception as exc:
                logger.warning("Пропускаю страницу %d в %s: %s", page_number, path.name, exc)
                continue
            if text and text.strip():
                sections.append((text, f"{path.name}#page={page_number}"))
    finally:
        doc.close()

    if not sections:
        logger.warning("Пропускаю %s: PDF пуст или не содержит извлекаемого текста", path.name)
    return sections


def _read_excel(path: Path) -> list[tuple[str, str]]:
    """Возвращает список (текст_листа, человекочитаемый_source) по каждому листу Excel."""
    sections: list[tuple[str, str]] = []
    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:
        logger.warning("Пропускаю %s: не удалось открыть Excel (%s)", path.name, exc)
        return sections

    for sheet_name in workbook.sheet_names:
        try:
            df = workbook.parse(sheet_name)
        except Exception as exc:
            logger.warning("Пропускаю лист %r в %s: %s", sheet_name, path.name, exc)
            continue
        if df.empty:
            continue
        text = df.to_csv(index=False)
        if text.strip():
            sections.append((text, f"{path.name}#sheet={sheet_name}"))

    if not sections:
        logger.warning("Пропускаю %s: Excel пуст или без данных", path.name)
    return sections


def _read_csv(path: Path) -> list[tuple[str, str]]:
    """Возвращает список из одного элемента (текст_csv, human_readable_source)."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Пропускаю %s: не удалось прочитать CSV (%s)", path.name, exc)
        return []
    if df.empty:
        logger.warning("Пропускаю %s: CSV пуст", path.name)
        return []
    text = df.to_csv(index=False)
    return [(text, path.name)] if text.strip() else []


def _read_txt(path: Path) -> list[tuple[str, str]]:
    """Возвращает список из одного элемента (текст, human_readable_source)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("Пропускаю %s: не удалось прочитать TXT (%s)", path.name, exc)
        return []
    if not text.strip():
        logger.warning("Пропускаю %s: файл пуст", path.name)
        return []
    return [(text, path.name)]


def _extract_sections(path: Path) -> list[tuple[str, str]]:
    """Диспетчер по расширению файла: путь -> список (текст, human_readable_source)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".xlsx", ".xls"}:
        return _read_excel(path)
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".txt":
        return _read_txt(path)
    return []


def _build_chunks(path: Path) -> list[Chunk]:
    """Читает файл, режет на чанки и оборачивает каждый в Chunk (app/state.py)."""
    sections = _extract_sections(path)
    chunks: list[Chunk] = []
    chunk_index = 0
    for section_text, human_source in sections:
        for piece in chunk_text(section_text):
            source_id = f"{path.stem}#{chunk_index}"
            chunks.append(
                Chunk(
                    text=piece,
                    source=human_source,
                    source_id=source_id,
                    metadata={"file": path.name},
                )
            )
            chunk_index += 1

    if not chunks:
        logger.warning("Пропускаю %s: не удалось получить ни одного чанка", path.name)
    return chunks


def ingest(folder: str) -> None:
    """
    Индексирует все поддерживаемые файлы из folder в локальный ChromaDB.

    Поддерживаемые форматы: .pdf, .xlsx, .xls, .csv, .txt. Битые или пустые
    файлы логируются и пропускаются — функция не падает и обрабатывает
    остальные файлы. Печатает итоговое число проиндексированных чанков.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        logger.error("Папка не найдена: %s", folder)
        print(f"Ошибка: папка не найдена: {folder}")
        return

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=HashingEmbedding())

    files = sorted(p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        logger.warning(
            "В папке %s не найдено поддерживаемых файлов (%s)",
            folder,
            ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )
        print("Проиндексировано чанков: 0")
        return

    total_chunks = 0
    for path in files:
        chunks = _build_chunks(path)
        if not chunks:
            continue

        collection.upsert(
            ids=[c.source_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "source_id": c.source_id, **c.metadata} for c in chunks],
        )
        total_chunks += len(chunks)
        logger.info("%s: проиндексировано %d чанков", path.name, len(chunks))

    logger.info("Готово. Файлов обработано: %d, всего чанков в индексе: %d", len(files), total_chunks)
    print(f"Проиндексировано чанков: {total_chunks}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python -m app.ingest <папка с файлами>")
        sys.exit(1)
    ingest(sys.argv[1])

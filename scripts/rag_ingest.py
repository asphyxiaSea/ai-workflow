from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.settings import (
    OLLAMA_BASE_URL,
    RAG_CHROMA_COLLECTION,
    RAG_CHROMA_PERSIST_DIR,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_DEFAULT_KNOWLEDGE_DOMAIN,
    RAG_EMBEDDING_BASE_URL,
    RAG_EMBEDDING_MODEL,
)


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _build_chunks(path: Path, *, knowledge_domain: str, book_id: str) -> list[Document]:
    text = _load_text(path)
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )
    source_doc = Document(
        page_content=text,
        metadata={
            "source": path.name,
            "path": str(path),
            "domain": knowledge_domain,
            "book_id": book_id,
        },
    )
    split_docs = splitter.split_documents([source_doc])

    chunks: list[Document] = []
    for idx, doc in enumerate(split_docs):
        metadata = dict(doc.metadata)
        metadata["chunk_index"] = idx
        chunks.append(Document(page_content=doc.page_content, metadata=metadata))
    return chunks


def _resolve_book_id(path: Path, custom_book_id: str | None) -> str:
    if custom_book_id:
        return custom_book_id
    return path.stem


def _get_store(collection_name: str | None) -> Chroma:
    embedding = OllamaEmbeddings(
        model=RAG_EMBEDDING_MODEL,
        base_url=RAG_EMBEDDING_BASE_URL or OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name=collection_name or RAG_CHROMA_COLLECTION,
        embedding_function=embedding,
        persist_directory=RAG_CHROMA_PERSIST_DIR,
    )


def _iter_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("Input file type is not supported. Use .txt/.md/.pdf")
        return [input_path]

    files: list[Path] = []
    for path in input_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files)


async def _ingest_files(
    paths: list[Path],
    collection_name: str | None,
    knowledge_domain: str,
    book_id: str | None,
) -> None:
    store = _get_store(collection_name)
    total_chunks = 0
    total_inserted = 0

    for path in paths:
        resolved_book_id = _resolve_book_id(path, book_id)
        chunks = await asyncio.to_thread(
            _build_chunks,
            path,
            knowledge_domain=knowledge_domain,
            book_id=resolved_book_id,
        )
        chunk_count = len(chunks)
        if chunk_count == 0:
            inserted_count = 0
            print(f"[SKIP] {path}: empty content")
        else:
            inserted_ids = await asyncio.to_thread(store.add_documents, chunks)
            inserted_count = len(inserted_ids)

        total_chunks += chunk_count
        total_inserted += inserted_count
        print(
            f"[OK] {path}: chunk_count={chunk_count}, inserted_count={inserted_count}, "
            f"collection={collection_name or RAG_CHROMA_COLLECTION}, "
            f"domain={knowledge_domain}, book_id={resolved_book_id}"
        )

    await asyncio.to_thread(store.persist)
    print(
        f"Done. files={len(paths)}, total_chunks={total_chunks}, total_inserted={total_inserted}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ingest files into Chroma for RAG")
    parser.add_argument("--input", required=True, help="A file or a directory path")
    parser.add_argument("--collection", default=None, help="Optional Chroma collection name")
    parser.add_argument(
        "--domain",
        default=RAG_DEFAULT_KNOWLEDGE_DOMAIN,
        help="Knowledge domain metadata for retrieval filtering",
    )
    parser.add_argument(
        "--book-id",
        default=None,
        help="Optional book id metadata. If omitted, each file uses its stem as book_id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    paths = _iter_files(input_path)
    if not paths:
        raise ValueError("No supported files found. Use .txt/.md/.pdf")

    asyncio.run(_ingest_files(paths, args.collection, args.domain, args.book_id))


if __name__ == "__main__":
    main()

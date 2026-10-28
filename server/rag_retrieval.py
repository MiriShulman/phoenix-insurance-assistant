"""Local semantic retrieval over Phoenix corpus chunks, with TF-IDF fallback.

FastEmbed creates multilingual embeddings locally; Chroma persists the vector
index on disk. If either dependency/model is unavailable, TF-IDF remains usable.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS_PATH = Path(__file__).parent / "data" / "pages_crawled_clean.jsonl"
MAX_CHUNK_CHARS = 1100
CHUNK_OVERLAP_CHARS = 160

_sentence_splitter = re.compile(r"(?<=[.!?׃])\s+|[\r\n]+")
_whitespace = re.compile(r"\s+")


def _normalize(value: str) -> str:
    return _whitespace.sub(
        " ",
        (value or "").replace("׳", "'").replace("’", "'").replace("״", '"')
    ).strip()


def _split_long_piece(
    piece: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    piece = _normalize(piece)
    if len(piece) <= max_chars:
        return [piece] if piece else []

    result: list[str] = []
    start = 0

    while start < len(piece):
        end = min(start + max_chars, len(piece))

        if end < len(piece):
            # Prefer a word boundary close to the target size.
            boundary = piece.rfind(" ", start + int(max_chars * 0.65), end)
            if boundary > start:
                end = boundary

        part = piece[start:end].strip()
        if part:
            result.append(part)

        if end >= len(piece):
            break

        start = max(start + 1, end - overlap)

    return result


def split_into_chunks(text: str) -> list[str]:
    """Split a page into overlapping, readable chunks while preserving sentences."""
    normalized = _normalize(text)
    if not normalized:
        return []

    pieces: list[str] = []

    for raw_piece in _sentence_splitter.split(normalized):
        raw_piece = _normalize(raw_piece)
        if not raw_piece:
            continue
        pieces.extend(_split_long_piece(raw_piece))

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece

        if len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate
            continue

        if current:
            chunks.append(current)

        # Very long pieces are already split above, but keep this guard anyway.
        subpieces = _split_long_piece(piece)

        if len(subpieces) > 1:
            chunks.extend(subpieces[:-1])
            current = subpieces[-1]
        else:
            current = piece

    if current:
        chunks.append(current)

    # Remove exact duplicate chunks while preserving order.
    unique: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        key = _normalize(chunk).casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(chunk)

    return unique


class LocalTfidfRetriever:
    """Cached word + character TF-IDF retriever; no API key or hosted service needed."""

    def __init__(self) -> None:
        self._signature: tuple[Any, ...] | None = None
        self._chunks: list[dict[str, Any]] = []
        self._word_vectorizer: TfidfVectorizer | None = None
        self._char_vectorizer: TfidfVectorizer | None = None
        self._word_matrix = None
        self._char_matrix = None

    @staticmethod
    def _corpus_signature(pages: list[dict[str, Any]]) -> tuple[Any, ...]:
        try:
            stat = CORPUS_PATH.stat()
            return (
                str(CORPUS_PATH.resolve()),
                stat.st_mtime_ns,
                stat.st_size,
                len(pages),
                sum(len(str(page.get("text", ""))) for page in pages),
            )
        except OSError:
            return (
                len(pages),
                sum(len(str(p.get("text", ""))) for p in pages),
                str(pages[0].get("url", "")) if pages else "",
                str(pages[-1].get("url", "")) if pages else "",
            )

    def _build(
        self,
        pages: list[dict[str, Any]],
        signature: tuple[Any, ...],
    ) -> None:
        chunks: list[dict[str, Any]] = []

        for page in pages:
            title = _normalize(str(page.get("title", "")))
            url = str(page.get("url", "")).strip()
            text = _normalize(str(page.get("text", "")))

            if not url or not text:
                continue

            for index, chunk_text in enumerate(split_into_chunks(text)):
                # Including the title in the indexed text helps identify the right page,
                # while keeping the actual citation excerpt limited to the chunk body.
                chunks.append({
                    "title": title or "עמוד באתר הפניקס",
                    "url": url,
                    "text": chunk_text,
                    "index": index,
                    "collected_at": page.get("collected_at"),
                    "indexed_text": f"{title} {chunk_text}".strip(),
                })

        if not chunks:
            self._chunks = []
            self._word_vectorizer = None
            self._char_vectorizer = None
            self._word_matrix = None
            self._char_matrix = None
            self._signature = signature
            return

        documents = [item["indexed_text"] for item in chunks]

        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
            sublinear_tf=True,
            max_features=40000,
            dtype=np.float32,
        )

        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
            max_features=80000,
            dtype=np.float32,
        )

        self._word_matrix = word_vectorizer.fit_transform(documents)
        self._char_matrix = char_vectorizer.fit_transform(documents)
        self._word_vectorizer = word_vectorizer
        self._char_vectorizer = char_vectorizer
        self._chunks = chunks
        self._signature = signature

    def search(
        self,
        question: str,
        pages: list[dict[str, Any]],
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        question = _normalize(question)

        if not question or not pages:
            return []

        signature = self._corpus_signature(pages)

        if signature != self._signature:
            self._build(pages, signature)

        if (
            not self._chunks
            or self._word_vectorizer is None
            or self._char_vectorizer is None
        ):
            return []

        word_query = self._word_vectorizer.transform([question])
        char_query = self._char_vectorizer.transform([question])

        word_scores = (self._word_matrix @ word_query.T).toarray().ravel()
        char_scores = (self._char_matrix @ char_query.T).toarray().ravel()

        # Character n-grams tolerate Hebrew inflections and spelling variants;
        # word n-grams preserve more precise phrase matches.
        scores = (0.45 * word_scores) + (0.55 * char_scores)

        # Return the strongest chunk per URL first, so one long page does not occupy
        # every result slot. Keep the best chunk separately for citation/snippet use.
        best_by_url: dict[str, dict[str, Any]] = {}

        for idx, score in enumerate(scores):
            if score <= 0:
                continue

            item = self._chunks[idx]
            url = item["url"]

            if url not in best_by_url or score > best_by_url[url]["score"]:
                best_by_url[url] = {
                    "title": item["title"],
                    "url": url,
                    "text": item["text"],
                    "score": float(score),
                    "chunk_index": item["index"],
                    "collected_at": item.get("collected_at"),
                }

        return sorted(
            best_by_url.values(),
            key=lambda item: item["score"],
            reverse=True,
        )[:top_k]


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_CACHE_DIR = Path(__file__).parent / "data" / "fastembed_cache"
CHROMA_PATH = Path(__file__).parent / "data" / "chroma_db"
CHROMA_COLLECTION_NAME = "phoenix_semantic_chunks_v1"

logger = logging.getLogger(__name__)


class LocalSemanticRetriever:
    """FastEmbed + persistent Chroma retrieval, initialized only when first used."""

    def __init__(self) -> None:
        self._signature: tuple[Any, ...] | None = None
        self._embedding_model = None
        self._client = None
        self._collection = None
        self._chunks: list[dict[str, Any]] = []
        self._failed = False
        self._failure_logged = False

    def _get_model(self):
        if self._embedding_model is None:
            from fastembed import TextEmbedding

            EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            self._embedding_model = TextEmbedding(
                model_name=MODEL_NAME,
                cache_dir=str(EMBEDDING_CACHE_DIR),
            )

        return self._embedding_model

    @staticmethod
    def _build_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        for page_index, page in enumerate(pages):
            title = _normalize(str(page.get("title", ""))) or "עמוד באתר הפניקס"
            url = str(page.get("url", "")).strip()
            text = _normalize(str(page.get("text", "")))

            if not url or not text:
                continue

            for chunk_index, chunk_text in enumerate(split_into_chunks(text)):
                indexed_text = f"{title} {chunk_text}".strip()
                raw_id = f"{url}\0{page_index}\0{chunk_index}\0{chunk_text}"
                chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

                chunks.append({
                    "id": chunk_id,
                    "title": title,
                    "url": url,
                    "text": chunk_text,
                    "index": chunk_index,
                    "collected_at": page.get("collected_at"),
                    "indexed_text": indexed_text,
                })

        return chunks

    @staticmethod
    def _signature_key(signature: tuple[Any, ...]) -> str:
        return hashlib.sha256(
            repr((signature, MODEL_NAME)).encode("utf-8")
        ).hexdigest()

    def _build_vector_index(
        self,
        pages: list[dict[str, Any]],
        signature: tuple[Any, ...],
    ) -> None:
        import chromadb

        chunks = self._build_chunks(pages)
        self._chunks = chunks

        if not chunks:
            self._client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            self._collection = None
            self._signature = signature
            return

        model = self._get_model()

        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        signature_key = self._signature_key(signature)

        expected_metadata = {
            "hnsw:space": "cosine",
            "corpus_signature": signature_key,
            "embedding_model": MODEL_NAME,
        }

        try:
            collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        except Exception:
            collection = None

        should_rebuild = (
            collection is None
            or (collection.metadata or {}).get("corpus_signature") != signature_key
            or (collection.metadata or {}).get("embedding_model") != MODEL_NAME
            or collection.count() != len(chunks)
        )

        if should_rebuild:
            if collection is not None:
                client.delete_collection(name=CHROMA_COLLECTION_NAME)

            collection = client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata=expected_metadata,
            )

            batch_size = 64

            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]

                embeddings = list(
                    model.embed([item["indexed_text"] for item in batch])
                )

                metadatas: list[dict[str, Any]] = []

                for item in batch:
                    metadata: dict[str, Any] = {
                        "title": item["title"],
                        "url": item["url"],
                        "chunk_index": int(item["index"]),
                    }

                    if item.get("collected_at") is not None:
                        metadata["collected_at"] = str(item["collected_at"])

                    metadatas.append(metadata)

                collection.add(
                    ids=[item["id"] for item in batch],
                    embeddings=[
                        embedding.tolist() for embedding in embeddings
                    ],
                    documents=[item["text"] for item in batch],
                    metadatas=metadatas,
                )

        self._client = client
        self._collection = collection
        self._signature = signature

    def search(
        self,
        question: str,
        pages: list[dict[str, Any]],
        top_k: int = 6,
    ) -> list[dict[str, Any]] | None:
        """Return semantic results, or None if semantic retrieval is unavailable."""
        question = _normalize(question)

        if not question or not pages:
            return []

        if self._failed:
            return None

        try:
            signature = LocalTfidfRetriever._corpus_signature(pages)

            if signature != self._signature:
                self._build_vector_index(pages, signature)

            if not self._collection or not self._chunks:
                return []

            model = self._get_model()
            query_vectors = list(model.embed([question]))

            if not query_vectors:
                return []

            candidate_count = min(
                max(top_k * 8, top_k),
                self._collection.count(),
            )

            if candidate_count <= 0:
                return []

            result = self._collection.query(
                query_embeddings=[query_vectors[0].tolist()],
                n_results=candidate_count,
                include=["documents", "metadatas", "distances"],
            )

            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]

            best_by_url: dict[str, dict[str, Any]] = {}

            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
            ):
                if not metadata or not metadata.get("url"):
                    continue

                # Chroma cosine distance is 1 - cosine similarity.
                score = max(0.0, min(1.0, 1.0 - float(distance)))

                if score <= 0:
                    continue

                url = str(metadata["url"])

                item = {
                    "title": str(
                        metadata.get("title") or "עמוד באתר הפניקס"
                    ),
                    "url": url,
                    "text": str(document or ""),
                    "score": score,
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "collected_at": metadata.get("collected_at"),
                }

                if url not in best_by_url or score > best_by_url[url]["score"]:
                    best_by_url[url] = item

            return sorted(
                best_by_url.values(),
                key=lambda item: item["score"],
                reverse=True,
            )[:top_k]

        except Exception as exc:
            self._failed = True

            if not self._failure_logged:
                logger.warning(
                    "Semantic retrieval is unavailable; using TF-IDF fallback. "
                    "Reason: %s",
                    exc,
                )
                self._failure_logged = True

            return None


# Keep the original TF-IDF retriever as a robust fallback and for compatibility.
retriever = LocalTfidfRetriever()
semantic_retriever = LocalSemanticRetriever()


def retrieve_relevant_chunks(
    question: str,
    pages: list[dict[str, Any]],
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """Use semantic retrieval first, falling back to the existing TF-IDF search."""
    semantic_results = semantic_retriever.search(
        question,
        pages,
        top_k=top_k,
    )

    if semantic_results:
        return semantic_results

    # If semantic search is unavailable or has no useful matches, retain the prior behavior.
    return retriever.search(question, pages, top_k=top_k)
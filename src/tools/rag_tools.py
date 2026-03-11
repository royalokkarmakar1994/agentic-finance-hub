"""
RAG tool using ChromaDB for citation-grounded research.

Stores retrieved financial documents in a persistent vector store
and retrieves the most semantically relevant passages to ground
agent responses in verified source material.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any

from crewai.tools import BaseTool
from pydantic import Field

CHROMA_COLLECTION = "financial_documents"
TOP_K_RESULTS = 5


class RAGStoreTool(BaseTool):
    """Store a financial document snippet in the ChromaDB vector store.

    Accepts a title, URL, publication date, and content string, then
    embeds and persists it so it can be retrieved by RAGRetrieveTool.
    """

    name: str = "rag_store_document"
    description: str = (
        "Store a financial document or news article in the vector database for "
        "later citation-accurate retrieval. "
        "Input format: 'TITLE|||URL|||PUBLISHED_DATE|||CONTENT' "
        "(fields separated by triple pipes '|||'). "
        "Example: 'Apple Q4 Results|||https://example.com|||2025-11-01|||Revenue beat...'"
    )
    persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )

    def _run(self, document_input: str) -> str:
        """Parse input and store the document in ChromaDB."""
        parts = document_input.split("|||")
        if len(parts) < 4:
            return (
                "[ERROR] Invalid format. Expected: "
                "'TITLE|||URL|||PUBLISHED_DATE|||CONTENT'"
            )

        title, url, pub_date, content = (
            parts[0].strip(),
            parts[1].strip(),
            parts[2].strip(),
            "|||".join(parts[3:]).strip(),
        )

        collection = self._get_collection()
        doc_id = hashlib.md5(url.encode()).hexdigest()
        stored_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        collection.upsert(
            documents=[content],
            metadatas=[{
                "title": title,
                "url": url,
                "published_date": pub_date,
                "stored_at": stored_at,
            }],
            ids=[doc_id],
        )
        return f"[RAG] Stored: '{title}' (ID: {doc_id})"

    def _get_collection(self) -> Any:
        """Return (or create) the ChromaDB collection."""
        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                DefaultEmbeddingFunction,
            )
        except ImportError as exc:
            raise ImportError(
                "chromadb is required. Run: pip install chromadb"
            ) from exc

        client = chromadb.PersistentClient(path=self.persist_dir)
        return client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=DefaultEmbeddingFunction(),
        )


class RAGRetrieveTool(BaseTool):
    """Retrieve relevant financial documents from ChromaDB by semantic search.

    Returns the top-K most semantically similar stored passages along
    with full citation metadata (title, URL, publication date).
    """

    name: str = "rag_retrieve_documents"
    description: str = (
        "Retrieve the most relevant financial documents from the vector database "
        "using semantic search. Returns results with full citation metadata "
        "(title, URL, publication date) for citation-accurate reporting. "
        "Input should be a natural-language query string, e.g. "
        "'Apple revenue growth Q4 2025 earnings beat'."
    )
    persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )

    def _run(self, query: str) -> str:
        """Query ChromaDB and return formatted, cited results."""
        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                DefaultEmbeddingFunction,
            )
        except ImportError as exc:
            raise ImportError(
                "chromadb is required. Run: pip install chromadb"
            ) from exc

        client = chromadb.PersistentClient(path=self.persist_dir)

        try:
            collection = client.get_collection(
                name=CHROMA_COLLECTION,
                embedding_function=DefaultEmbeddingFunction(),
            )
        except Exception:
            return (
                "[RAG] No documents in the vector store yet. "
                "Use rag_store_document to add documents first."
            )

        results = collection.query(
            query_texts=[query],
            n_results=min(TOP_K_RESULTS, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return "[RAG] No relevant documents found for the query."

        lines = [f"[RAG RETRIEVAL] Query: '{query}'\n"]
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):
            relevance = round((1 - dist) * 100, 1)
            lines.append(
                f"[{i}] Relevance: {relevance}%\n"
                f"    Title  : {meta.get('title', 'N/A')}\n"
                f"    URL    : {meta.get('url', 'N/A')}\n"
                f"    Date   : {meta.get('published_date', 'N/A')}\n"
                f"    Excerpt: {doc[:400]}\n"
            )
        return "\n".join(lines)

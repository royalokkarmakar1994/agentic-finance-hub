"""
Unit tests for the RAG (ChromaDB) tools.

All ChromaDB calls are mocked so tests are fully offline.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestRAGStoreTool(unittest.TestCase):
    """Tests for RAGStoreTool._run()."""

    def _make_tool(self, persist_dir: str = "/tmp/test_chroma") -> object:
        from src.tools.rag_tools import RAGStoreTool
        return RAGStoreTool(persist_dir=persist_dir)

    def test_invalid_format_returns_error(self):
        tool = self._make_tool()
        result = tool._run("bad input without pipes")
        self.assertIn("ERROR", result)

    @patch("chromadb.PersistentClient")
    def test_valid_input_stores_and_returns_confirmation(self, mock_client_cls):
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        tool = self._make_tool()
        doc_input = (
            "Apple Q4|||https://reuters.com/apple|||2025-11-01|||Revenue beat estimates."
        )
        result = tool._run(doc_input)

        self.assertIn("Stored", result)
        mock_collection.upsert.assert_called_once()

    @patch("chromadb.PersistentClient")
    def test_content_with_pipes_is_preserved(self, mock_client_cls):
        """Content fields that contain '|||' should be merged back correctly."""
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        tool = self._make_tool()
        # Content contains '|||' as part of the text
        doc_input = (
            "Title|||https://example.com|||2025-11-01|||Part A|||Part B|||Part C"
        )
        result = tool._run(doc_input)
        self.assertIn("Stored", result)

        # Verify the full content (parts 3+) was passed to upsert
        call_args = mock_collection.upsert.call_args
        stored_document = call_args.kwargs.get("documents", [None])[0] or call_args[1].get("documents", [None])[0]
        self.assertIn("Part A", stored_document)
        self.assertIn("Part B", stored_document)


class TestRAGRetrieveTool(unittest.TestCase):
    """Tests for RAGRetrieveTool._run()."""

    def _make_tool(self, persist_dir: str = "/tmp/test_chroma") -> object:
        from src.tools.rag_tools import RAGRetrieveTool
        return RAGRetrieveTool(persist_dir=persist_dir)

    @patch("chromadb.PersistentClient")
    def test_empty_collection_returns_no_docs_message(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client_cls.return_value = mock_client

        tool = self._make_tool()
        result = tool._run("Apple earnings")
        self.assertIn("No documents", result)

    @patch("chromadb.PersistentClient")
    def test_retrieval_returns_citations(self, mock_client_cls):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["Revenue grew 8% year-over-year."]],
            "metadatas": [[{
                "title": "Apple Q4 Results",
                "url": "https://reuters.com/aapl",
                "published_date": "2025-11-01",
            }]],
            "distances": [[0.15]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        tool = self._make_tool()
        result = tool._run("Apple revenue")

        self.assertIn("Apple Q4 Results", result)
        self.assertIn("reuters.com", result)
        self.assertIn("2025-11-01", result)
        self.assertIn("Revenue grew", result)

    @patch("chromadb.PersistentClient")
    def test_no_results_message(self, mock_client_cls):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        tool = self._make_tool()
        result = tool._run("unrelated query")
        self.assertIn("No relevant documents", result)


if __name__ == "__main__":
    unittest.main()

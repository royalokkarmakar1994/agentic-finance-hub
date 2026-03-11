"""
Unit tests for the Tavily search tool.

All network calls are mocked so tests are fully offline.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.tools.search_tools import TavilyFinancialSearchTool


def _make_result(title: str, url: str, content: str, days_ago: int) -> dict:
    pub_date = (datetime.now(tz=timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "title": title,
        "url": url,
        "content": content,
        "published_date": pub_date,
    }


class TestTavilySearchFreshness(unittest.TestCase):
    """Test the _check_freshness helper logic."""

    def setUp(self):
        self.tool = TavilyFinancialSearchTool(api_key="test-key")
        self.cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)

    def test_fresh_article(self):
        pub = (datetime.now(tz=timezone.utc) - timedelta(days=2)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.assertEqual(self.tool._check_freshness(pub, self.cutoff), "FRESH")

    def test_stale_article(self):
        pub = (datetime.now(tz=timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.assertEqual(self.tool._check_freshness(pub, self.cutoff), "STALE")

    def test_missing_date(self):
        self.assertEqual(self.tool._check_freshness("", self.cutoff), "DATE_UNKNOWN")

    def test_malformed_date(self):
        self.assertEqual(
            self.tool._check_freshness("not-a-date", self.cutoff), "DATE_UNKNOWN"
        )


class TestTavilyFormatResults(unittest.TestCase):
    """Test the _format_results output structure."""

    def setUp(self):
        self.tool = TavilyFinancialSearchTool(api_key="test-key")
        self.cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)
        self.current_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_output_contains_headers(self):
        response = {
            "answer": "Test summary",
            "results": [
                _make_result("Fresh News", "https://example.com", "Content here", 2)
            ],
        }
        output = self.tool._format_results(response, self.cutoff, self.current_ts)
        self.assertIn("SEARCH RESULTS", output)
        self.assertIn("STALENESS GUARD", output)
        self.assertIn("FRESH", output)
        self.assertIn("Fresh News", output)

    def test_stale_result_labelled(self):
        response = {
            "answer": None,
            "results": [
                _make_result("Old News", "https://example.com", "Old content", 30)
            ],
        }
        output = self.tool._format_results(response, self.cutoff, self.current_ts)
        self.assertIn("STALE", output)


class TestTavilyRunMocked(unittest.TestCase):
    """Test the _run method with a fully mocked TavilyClient."""

    @patch("tavily.TavilyClient")
    def test_run_returns_formatted_string(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "Apple Q4 results beat expectations.",
            "results": [
                _make_result(
                    "Apple beats Q4", "https://reuters.com/aapl", "Revenue up 8%", 1
                )
            ],
        }
        mock_client_cls.return_value = mock_client

        tool = TavilyFinancialSearchTool(api_key="test-key")
        result = tool._run("AAPL Q4 2025 earnings")

        self.assertIn("SEARCH RESULTS", result)
        self.assertIn("Apple beats Q4", result)
        mock_client.search.assert_called_once()

    def test_missing_api_key_raises(self):
        tool = TavilyFinancialSearchTool(api_key="")
        with self.assertRaises(ValueError):
            tool._run("AAPL")


if __name__ == "__main__":
    unittest.main()

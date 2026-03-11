"""
Tests for the crew orchestration and JSON report parsing.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from src.crew import _parse_json_output


def _make_mock_llm() -> object:
    """Return a BaseLLM subclass instance that satisfies CrewAI validation."""
    from crewai.llms.base_llm import BaseLLM

    class _MockLLM(BaseLLM):
        def __init__(self):  # noqa: D107
            pass  # skip parent __init__ – no API key needed

        def call(self, *args, **kwargs):  # noqa: D102
            return "mock response"

        def get_context_window_size(self) -> int:  # noqa: D102
            return 128_000

        def supports_function_calling(self) -> bool:  # noqa: D102
            return True

        def supports_stop_words(self) -> bool:  # noqa: D102
            return True

        def get_supported_params(self) -> list:  # noqa: D102
            return []

    return _MockLLM()


class TestParseJsonOutput(unittest.TestCase):
    """Unit tests for _parse_json_output helper."""

    def test_valid_json_parsed(self):
        report = {
            "ticker": "AAPL",
            "confidence_score": 75,
            "news_sentiment": "POSITIVE",
        }
        result = _parse_json_output(json.dumps(report), "AAPL")
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["confidence_score"], 75)

    def test_json_with_markdown_fences(self):
        report = {"ticker": "MSFT", "confidence_score": 60}
        raw = f"```json\n{json.dumps(report)}\n```"
        result = _parse_json_output(raw, "MSFT")
        self.assertEqual(result["confidence_score"], 60)

    def test_json_embedded_in_text(self):
        report = {"ticker": "GOOGL", "confidence_score": 50}
        raw = f"Here is the result:\n{json.dumps(report)}\nEnd of report."
        result = _parse_json_output(raw, "GOOGL")
        self.assertEqual(result["confidence_score"], 50)

    def test_invalid_json_returns_error_dict(self):
        result = _parse_json_output("This is not JSON at all.", "TSLA")
        self.assertIn("error", result)
        self.assertEqual(result["ticker"], "TSLA")
        self.assertIn("raw_output", result)

    def test_empty_string_returns_error_dict(self):
        result = _parse_json_output("", "AMZN")
        self.assertIn("error", result)


class TestRunAnalysisValidation(unittest.TestCase):
    """Test input validation in run_analysis."""

    def test_empty_ticker_raises(self):
        from src.crew import run_analysis
        with self.assertRaises(ValueError):
            run_analysis("   ")


class TestBuildCrewStructure(unittest.TestCase):
    """Verify that build_crew creates a properly wired crew."""

    @patch("src.crew._build_llm")
    def test_crew_has_three_agents(self, mock_llm_fn):
        mock_llm_fn.return_value = _make_mock_llm()
        from src.crew import build_crew
        crew = build_crew("AAPL")
        self.assertEqual(len(crew.agents), 3)

    @patch("src.crew._build_llm")
    def test_crew_has_three_tasks(self, mock_llm_fn):
        mock_llm_fn.return_value = _make_mock_llm()
        from src.crew import build_crew
        crew = build_crew("AAPL")
        self.assertEqual(len(crew.tasks), 3)


if __name__ == "__main__":
    unittest.main()

"""
Tavily Search tool with noise-filtering for financial news.

Wraps the Tavily API to retrieve recent, domain-verified financial
news while enforcing a staleness guard so only articles published
within MAX_AGE_DAYS are accepted by the self-correction loop.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from crewai.tools import BaseTool
from pydantic import Field

MAX_AGE_DAYS = 7
MAX_RESULTS = 10
FINANCIAL_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "marketwatch.com",
    "seekingalpha.com",
    "finance.yahoo.com",
    "sec.gov",
    "businesswire.com",
    "prnewswire.com",
    "fool.com",
    "investopedia.com",
    "barrons.com",
    "thestreet.com",
]


class TavilyFinancialSearchTool(BaseTool):
    """Search for recent financial news using the Tavily API.

    Returns a list of verified, date-stamped results restricted to
    trusted financial domains. Articles older than MAX_AGE_DAYS are
    flagged so the self-correction loop can discard stale data.
    """

    name: str = "tavily_financial_search"
    description: str = (
        "Search for the latest financial news, analyst reports, and SEC filings "
        "for a given stock ticker or company name. Returns structured results "
        "with publication dates so stale information can be filtered out. "
        "Input should be a search query string such as 'AAPL earnings Q4 2025'."
    )
    api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    def _run(self, query: str) -> str:
        """Execute a financial news search and return formatted results."""
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise ImportError(
                "tavily-python is required. Run: pip install tavily-python"
            ) from exc

        if not self.api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable is not set. "
                "Please add it to your .env file."
            )

        client = TavilyClient(api_key=self.api_key)
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=MAX_AGE_DAYS)
        current_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        response = client.search(
            query=query,
            search_depth="advanced",
            include_domains=FINANCIAL_DOMAINS,
            max_results=MAX_RESULTS,
            include_answer=True,
            include_raw_content=False,
        )

        results = self._format_results(response, cutoff, current_ts)
        return results

    def _format_results(
        self,
        response: dict[str, Any],
        cutoff: datetime,
        current_ts: str,
    ) -> str:
        """Format and date-validate Tavily response results."""
        lines: list[str] = [
            f"[SEARCH RESULTS] Current market timestamp: {current_ts}",
            f"[STALENESS GUARD] Only articles published after "
            f"{cutoff.strftime('%Y-%m-%d')} are marked FRESH.\n",
        ]

        if response.get("answer"):
            lines.append(f"Summary: {response['answer']}\n")

        for i, result in enumerate(response.get("results", []), start=1):
            pub_date_str = result.get("published_date", "")
            freshness = self._check_freshness(pub_date_str, cutoff)
            lines.append(
                f"[{i}] [{freshness}] {result.get('title', 'No title')}\n"
                f"    URL: {result.get('url', '')}\n"
                f"    Published: {pub_date_str or 'Unknown'}\n"
                f"    Snippet: {result.get('content', '')[:500]}\n"
            )

        return "\n".join(lines)

    @staticmethod
    def _check_freshness(pub_date_str: str, cutoff: datetime) -> str:
        """Return 'FRESH' or 'STALE' based on article publication date."""
        if not pub_date_str:
            return "DATE_UNKNOWN"
        try:
            pub_date = datetime.fromisoformat(
                pub_date_str.replace("Z", "+00:00")
            )
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            return "FRESH" if pub_date >= cutoff else "STALE"
        except (ValueError, TypeError):
            return "DATE_UNKNOWN"

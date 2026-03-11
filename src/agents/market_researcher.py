"""
Market Researcher agent definition.

Uses the Tavily Search API and ChromaDB RAG tools to gather
noise-filtered financial news, filings, and analyst sentiments
for a specified stock ticker.
"""

from __future__ import annotations

from crewai import Agent

from src.tools.rag_tools import RAGRetrieveTool, RAGStoreTool
from src.tools.search_tools import TavilyFinancialSearchTool


def create_market_researcher(llm: object) -> Agent:
    """Instantiate the Market Researcher agent.

    Args:
        llm: A CrewAI-compatible LLM instance (e.g. Gemini 1.5 Pro).

    Returns:
        A configured CrewAI Agent ready to be added to a Crew.
    """
    return Agent(
        role="Senior Equity Research Analyst",
        goal=(
            "Gather, verify, and synthesise the most recent financial news, "
            "SEC filings, and analyst sentiments for the target stock ticker. "
            "Every piece of information MUST include a publication date. "
            "Discard any article or snippet labelled STALE or DATE_UNKNOWN "
            "by the search tool. Store all FRESH results in the RAG vector "
            "store for citation-accurate downstream retrieval."
        ),
        backstory=(
            "You are a seasoned equity research analyst at a top-tier investment "
            "bank. You have spent 15 years reading 10-K filings, earnings call "
            "transcripts, and breaking market news. You are meticulous about "
            "source dates and never mix data from different fiscal years. "
            "Your reports are the gold standard for accuracy across the desk."
        ),
        tools=[
            TavilyFinancialSearchTool(),
            RAGStoreTool(),
            RAGRetrieveTool(),
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

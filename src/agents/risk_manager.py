"""
Risk Manager agent definition.

Acts as a quality-control critic that cross-references news sentiment
with technical indicators, eliminates contradictions and hallucinations,
and produces the final confidence-scored JSON equity research report.
"""

from __future__ import annotations

from crewai import Agent

from src.tools.rag_tools import RAGRetrieveTool


def create_risk_manager(llm: object) -> Agent:
    """Instantiate the Risk Manager (Critic) agent.

    Args:
        llm: A CrewAI-compatible LLM instance (e.g. Gemini 1.5 Pro).

    Returns:
        A configured CrewAI Agent ready to be added to a Crew.
    """
    return Agent(
        role="Chief Risk Officer & Research Quality Controller",
        goal=(
            "Cross-reference the Market Researcher's news sentiment findings "
            "against the Technical Analyst's price-action indicators. "
            "Identify and eliminate any contradictions, hallucinations, or "
            "data from incorrect fiscal periods. "
            "Produce a final, validated JSON report containing: ticker symbol, "
            "analysis timestamp, news_sentiment (POSITIVE / NEGATIVE / NEUTRAL), "
            "technical_bias (BULLISH / BEARISH / NEUTRAL), key_indicators dict, "
            "top_news_citations list (title + URL + published_date), "
            "risks list, confidence_score (0–100), and investment_thesis string. "
            "The confidence_score must reflect the degree of alignment between "
            "sentiment and technical data."
        ),
        backstory=(
            "You are the Chief Risk Officer of a quantitative hedge fund. You "
            "have seen every flavour of hallucination and data error in AI-generated "
            "research. Your sole purpose is to ensure that the final report is "
            "factually grounded, internally consistent, and free from stale data. "
            "If the news sentiment and technical signals contradict each other, "
            "you lower the confidence score accordingly and document the conflict. "
            "You always output strict, parseable JSON with no trailing commentary."
        ),
        tools=[RAGRetrieveTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

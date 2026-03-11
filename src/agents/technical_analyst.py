"""
Technical Analyst agent definition.

Uses the yfinance-powered TechnicalAnalysisTool to fetch real-time
price data and compute RSI, Moving Averages, and MACD signals for
a given stock ticker.
"""

from __future__ import annotations

from crewai import Agent

from src.tools.technical_tools import TechnicalAnalysisTool


def create_technical_analyst(llm: object) -> Agent:
    """Instantiate the Technical Analyst agent.

    Args:
        llm: A CrewAI-compatible LLM instance (e.g. Gemini 1.5 Pro).

    Returns:
        A configured CrewAI Agent ready to be added to a Crew.
    """
    return Agent(
        role="Quantitative Technical Analyst",
        goal=(
            "Retrieve live price-action data for the target ticker and compute "
            "all key technical indicators: RSI-14, SMA-20, SMA-50, EMA-12, "
            "EMA-26, and volume trend. Derive a clear overall technical bias "
            "(BULLISH / BEARISH / NEUTRAL) with supporting signal details. "
            "Your output will be used by the Risk Manager to correlate with "
            "news sentiment, so be precise and include exact numeric values."
        ),
        backstory=(
            "You are a quantitative analyst who specialises in technical market "
            "microstructure. You have built trading algorithms for hedge funds "
            "and know exactly how RSI divergence, moving-average crossovers, and "
            "volume spikes translate into actionable signals. You communicate "
            "findings with mathematical precision."
        ),
        tools=[TechnicalAnalysisTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

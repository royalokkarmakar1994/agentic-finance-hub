"""
CrewAI Crew orchestration for the Agentic Finance Hub.

Wires together the Market Researcher, Technical Analyst, and Risk Manager
agents into a hierarchical process where each agent's output feeds the
next, and the Manager LLM enforces the self-correction loop.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from crewai import Crew, LLM, Process, Task

from src.agents.market_researcher import create_market_researcher
from src.agents.risk_manager import create_risk_manager
from src.agents.technical_analyst import create_technical_analyst


def _build_llm() -> LLM:
    """Instantiate the Gemini 1.5 Pro LLM via Google AI Studio."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    return LLM(
        model=f"gemini/{model}",
        temperature=temperature,
        api_key=api_key,
    )


def _build_tasks(
    ticker: str,
    researcher: object,
    analyst: object,
    risk_mgr: object,
) -> list[Task]:
    """Create the ordered task chain for the three-agent workflow."""
    current_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    research_task = Task(
        description=(
            f"Perform a comprehensive financial news search for the stock ticker "
            f"'{ticker}' as of {current_ts}.\n\n"
            f"1. Use the tavily_financial_search tool with query "
            f"   '{ticker} stock news earnings analyst sentiment'.\n"
            f"2. CRITICAL: Inspect every result's publication date. "
            f"   Any result labelled STALE or DATE_UNKNOWN MUST be discarded.\n"
            f"3. Store each FRESH article using rag_store_document.\n"
            f"4. Summarise the verified news sentiment: overall tone "
            f"   (POSITIVE / NEGATIVE / NEUTRAL), key themes, and risks.\n"
            f"5. Include full citation for every fact: title, URL, and date."
        ),
        expected_output=(
            "A structured markdown summary containing:\n"
            "- Overall news sentiment (POSITIVE / NEGATIVE / NEUTRAL)\n"
            "- 3–5 key themes from fresh articles\n"
            "- Cited news list (title | URL | published_date)\n"
            "- Identified risk factors mentioned in the news"
        ),
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            f"Perform a full technical analysis of '{ticker}' using live "
            f"market data as of {current_ts}.\n\n"
            f"1. Use the technical_analysis tool with input '{ticker}'.\n"
            f"2. Report exact numeric values for RSI-14, SMA-20, SMA-50, "
            f"   EMA-12, EMA-26, current price, and volume trend.\n"
            f"3. State the overall technical bias and list every signal "
            f"   that contributed to it.\n"
            f"4. Note any significant divergence between price and volume."
        ),
        expected_output=(
            "A structured technical report containing:\n"
            "- Current price and daily change %\n"
            "- RSI-14 value and zone interpretation\n"
            "- SMA-20, SMA-50, EMA-12, EMA-26 values\n"
            "- Volume trend assessment\n"
            "- Overall technical bias (BULLISH / BEARISH / NEUTRAL) with reasoning"
        ),
        agent=analyst,
        context=[research_task],
    )

    risk_task = Task(
        description=(
            f"Synthesise the Market Researcher's news findings and the Technical "
            f"Analyst's indicator data for '{ticker}' into a final validated "
            f"JSON report.\n\n"
            f"Self-Correction Checklist (verify before outputting):\n"
            f"  ✓ All cited articles have a FRESH publication date.\n"
            f"  ✓ No financial figures from prior fiscal years are cited as current.\n"
            f"  ✓ News sentiment and technical bias are compared and any "
            f"    contradictions are documented.\n"
            f"  ✓ confidence_score reflects alignment: "
            f"    high (70-100) when both agree, medium (40-69) when mixed, "
            f"    low (0-39) when conflicting.\n\n"
            f"Output ONLY the JSON object – no markdown fences, no extra text."
        ),
        expected_output=(
            "A single valid JSON object with these exact keys:\n"
            "{\n"
            '  "ticker": "string",\n'
            '  "analysis_timestamp": "ISO-8601 string",\n'
            '  "news_sentiment": "POSITIVE|NEGATIVE|NEUTRAL",\n'
            '  "technical_bias": "BULLISH|BEARISH|NEUTRAL",\n'
            '  "key_indicators": {\n'
            '    "current_price": number,\n'
            '    "rsi_14": number,\n'
            '    "sma_20": number,\n'
            '    "sma_50": number,\n'
            '    "ema_12": number,\n'
            '    "ema_26": number,\n'
            '    "volume_trend": "string"\n'
            "  },\n"
            '  "top_news_citations": [\n'
            '    {"title": "string", "url": "string", "published_date": "string"}\n'
            "  ],\n"
            '  "risks": ["string"],\n'
            '  "confidence_score": number,\n'
            '  "investment_thesis": "string"\n'
            "}"
        ),
        agent=risk_mgr,
        context=[research_task, analysis_task],
    )

    return [research_task, analysis_task, risk_task]


def build_crew(ticker: str) -> Crew:
    """Build and return the hierarchical CrewAI Crew for a given ticker.

    Args:
        ticker: The stock ticker symbol to analyse, e.g. 'AAPL'.

    Returns:
        A fully configured Crew instance ready to be kicked off.
    """
    llm = _build_llm()
    researcher = create_market_researcher(llm)
    analyst = create_technical_analyst(llm)
    risk_mgr = create_risk_manager(llm)

    tasks = _build_tasks(ticker, researcher, analyst, risk_mgr)

    verbose = os.getenv("VERBOSE", "true").lower() == "true"

    return Crew(
        agents=[researcher, analyst, risk_mgr],
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
        manager_llm=llm,
    )


def run_analysis(ticker: str) -> dict[str, Any]:
    """Run the full multi-agent analysis pipeline for a stock ticker.

    Args:
        ticker: The stock ticker symbol to analyse, e.g. 'AAPL'.

    Returns:
        A dictionary containing the validated JSON report, or an error
        dict if parsing fails.

    Raises:
        ValueError: If ticker is empty or the LLM key is missing.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker symbol must not be empty.")

    crew = build_crew(ticker)
    raw_output = crew.kickoff()

    result_str = (
        raw_output.raw
        if hasattr(raw_output, "raw")
        else str(raw_output)
    )

    return _parse_json_output(result_str, ticker)


def _parse_json_output(raw: str, ticker: str) -> dict[str, Any]:
    """Extract and parse the JSON report from the agent's raw output."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON object within the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return {
        "ticker": ticker,
        "error": "Failed to parse JSON from agent output.",
        "raw_output": raw[:2000],
        "analysis_timestamp": datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }

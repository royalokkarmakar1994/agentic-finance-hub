"""
Entry point for the Agentic Finance Hub.

Usage:
    python -m src.main AAPL
    python -m src.main MSFT --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env variables before importing any API-dependent modules
load_dotenv()

from src.crew import run_analysis  # noqa: E402  (import after load_dotenv)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentic-finance-hub",
        description=(
            "Autonomous multi-agent equity research system. "
            "Analyses a stock ticker using live news and technical indicators "
            "to produce a confidence-scored JSON report."
        ),
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol to analyse (e.g. AAPL, MSFT, GOOGL).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save the JSON report (e.g. report.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the analysis pipeline and print/save the result."""
    args = _parse_args(argv)
    ticker = args.ticker.strip().upper()

    print(f"\n{'='*60}")
    print(f"  Agentic Finance Hub — Analysing: {ticker}")
    print(f"{'='*60}\n")

    try:
        report = run_analysis(ticker)
    except ValueError as exc:
        print(f"[ERROR] Configuration error: {exc}", file=sys.stderr)
        return 1

    output_str = json.dumps(report, indent=2)
    print("\n" + "="*60)
    print("  FINAL REPORT")
    print("="*60)
    print(output_str)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_str, encoding="utf-8")
        print(f"\n[INFO] Report saved to: {output_path.resolve()}")

    confidence = report.get("confidence_score")
    if isinstance(confidence, (int, float)):
        print(f"\n[RESULT] Confidence Score for {ticker}: {confidence}/100")

    return 0


if __name__ == "__main__":
    sys.exit(main())

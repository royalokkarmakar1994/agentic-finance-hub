"""
Technical analysis tool using yfinance.

Fetches real-time price data and calculates RSI, Simple Moving Averages,
EMA, volume trend, and derives a technical bias signal for a given ticker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from crewai.tools import BaseTool


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate the Relative Strength Index (RSI) for the last data point."""
    delta = series.diff().dropna()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()

    # When avg_loss is zero, all movement is upward → RSI = 100
    rsi = np.where(avg_loss == 0, 100.0, 100 - (100 / (1 + avg_gain / avg_loss)))
    rsi = pd.Series(rsi, index=avg_gain.index)
    return round(float(rsi.iloc[-1]), 2)


def _compute_ma(series: pd.Series, window: int) -> float:
    """Return the Simple Moving Average for the last data point."""
    if len(series) < window:
        return float("nan")
    return round(float(series.rolling(window=window).mean().iloc[-1]), 4)


def _compute_ema(series: pd.Series, span: int) -> float:
    """Return the Exponential Moving Average for the last data point."""
    if len(series) < span:
        return float("nan")
    return round(float(series.ewm(span=span, adjust=False).mean().iloc[-1]), 4)


def _interpret_rsi(rsi: float) -> str:
    """Translate RSI value into a human-readable zone description."""
    if rsi >= 70:
        return "Overbought (≥70)"
    if rsi <= 30:
        return "Oversold (≤30)"
    return "Neutral (30–70)"


def _volume_trend(volume: pd.Series, window: int = 20) -> str:
    """Compare latest volume to its rolling average."""
    if len(volume) < window:
        return "Insufficient data"
    avg_vol = volume.rolling(window=window).mean().iloc[-1]
    latest_vol = volume.iloc[-1]
    ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
    if ratio >= 1.5:
        return f"HIGH ({ratio:.1f}x avg)"
    if ratio <= 0.5:
        return f"LOW ({ratio:.1f}x avg)"
    return f"NORMAL ({ratio:.1f}x avg)"


def _derive_technical_bias(
    price: float,
    sma_20: float,
    sma_50: float,
    ema_12: float,
    ema_26: float,
    rsi: float,
) -> dict[str, Any]:
    """Synthesise individual indicators into an overall technical bias."""
    bullish_signals = 0
    bearish_signals = 0
    signals: list[str] = []

    if not np.isnan(sma_20) and price > sma_20:
        bullish_signals += 1
        signals.append("Price above SMA-20 (bullish)")
    elif not np.isnan(sma_20):
        bearish_signals += 1
        signals.append("Price below SMA-20 (bearish)")

    if not np.isnan(sma_50) and price > sma_50:
        bullish_signals += 1
        signals.append("Price above SMA-50 (bullish)")
    elif not np.isnan(sma_50):
        bearish_signals += 1
        signals.append("Price below SMA-50 (bearish)")

    if not (np.isnan(ema_12) or np.isnan(ema_26)) and ema_12 > ema_26:
        bullish_signals += 1
        signals.append("EMA-12 above EMA-26 / MACD positive (bullish)")
    elif not (np.isnan(ema_12) or np.isnan(ema_26)):
        bearish_signals += 1
        signals.append("EMA-12 below EMA-26 / MACD negative (bearish)")

    if rsi < 30:
        bullish_signals += 1
        signals.append("RSI oversold – potential reversal upward (bullish)")
    elif rsi > 70:
        bearish_signals += 1
        signals.append("RSI overbought – potential reversal downward (bearish)")

    total = bullish_signals + bearish_signals
    if total == 0:
        bias = "NEUTRAL"
        strength = 0
    else:
        strength = round((bullish_signals - bearish_signals) / total * 100, 1)
        if strength > 25:
            bias = "BULLISH"
        elif strength < -25:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

    return {
        "bias": bias,
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "strength_pct": strength,
        "signal_details": signals,
    }


class TechnicalAnalysisTool(BaseTool):
    """Fetch live market data and compute technical indicators via yfinance.

    Calculates RSI-14, SMA-20, SMA-50, EMA-12, EMA-26, and volume
    trend, then returns a structured technical report for a ticker.
    """

    name: str = "technical_analysis"
    description: str = (
        "Fetch real-time stock price data and compute key technical indicators "
        "(RSI-14, SMA-20, SMA-50, EMA-12/26, volume trend) for a given ticker. "
        "Returns a structured technical analysis report with an overall bias "
        "(BULLISH / BEARISH / NEUTRAL). "
        "Input must be a valid stock ticker symbol, e.g. 'AAPL' or 'MSFT'."
    )

    def _run(self, ticker: str) -> str:
        """Download price data and return formatted technical analysis."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required. Run: pip install yfinance"
            ) from exc

        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo", interval="1d", auto_adjust=True)

        if hist.empty:
            return (
                f"[ERROR] No price data found for '{ticker}'. "
                "Please verify the ticker symbol."
            )

        close = hist["Close"]
        volume = hist["Volume"]

        rsi = _compute_rsi(close)
        sma_20 = _compute_ma(close, 20)
        sma_50 = _compute_ma(close, 50)
        ema_12 = _compute_ema(close, 12)
        ema_26 = _compute_ema(close, 26)

        current_price = round(float(close.iloc[-1]), 4)
        prev_price = round(float(close.iloc[-2]), 4) if len(close) >= 2 else current_price
        price_change_pct = round((current_price - prev_price) / prev_price * 100, 2)

        vol_trend = _volume_trend(volume)
        bias_data = _derive_technical_bias(
            current_price, sma_20, sma_50, ema_12, ema_26, rsi
        )

        period_high = round(float(close.max()), 4)
        period_low = round(float(close.min()), 4)
        as_of = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        lines = [
            f"[TECHNICAL ANALYSIS] Ticker: {ticker} | As of: {as_of}",
            "",
            f"  Current Price   : ${current_price}",
            f"  Daily Change    : {price_change_pct:+.2f}%",
            f"  90-Day High     : ${period_high}",
            f"  90-Day Low      : ${period_low}",
            "",
            "  --- Key Indicators ---",
            f"  RSI-14          : {rsi} ({_interpret_rsi(rsi)})",
            f"  SMA-20          : ${sma_20}",
            f"  SMA-50          : ${sma_50}",
            f"  EMA-12          : ${ema_12}",
            f"  EMA-26          : ${ema_26}",
            f"  Volume Trend    : {vol_trend}",
            "",
            "  --- Technical Bias ---",
            f"  Overall Bias    : {bias_data['bias']}",
            f"  Strength        : {bias_data['strength_pct']:+.1f}%",
            f"  Bullish Signals : {bias_data['bullish_signals']}",
            f"  Bearish Signals : {bias_data['bearish_signals']}",
            "",
            "  Signal Details:",
        ]
        for detail in bias_data["signal_details"]:
            lines.append(f"    • {detail}")

        return "\n".join(lines)

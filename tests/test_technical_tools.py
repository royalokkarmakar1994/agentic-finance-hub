"""
Unit tests for the technical analysis tool.

These tests use synthetic price data so no network calls are made.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.tools.technical_tools import (
    TechnicalAnalysisTool,
    _compute_ema,
    _compute_ma,
    _compute_rsi,
    _derive_technical_bias,
    _interpret_rsi,
    _volume_trend,
)


class TestComputeRSI(unittest.TestCase):
    def test_overbought(self):
        """Steadily rising prices should produce an RSI > 70."""
        prices = pd.Series([float(i) for i in range(100, 130)])
        rsi = _compute_rsi(prices)
        self.assertGreater(rsi, 70)

    def test_oversold(self):
        """Steadily falling prices should produce an RSI < 30."""
        prices = pd.Series([float(i) for i in range(130, 100, -1)])
        rsi = _compute_rsi(prices)
        self.assertLess(rsi, 30)

    def test_neutral(self):
        """Alternating up/down prices should produce an RSI near 50."""
        prices = pd.Series([100.0 + (i % 2) * 2 for i in range(30)])
        rsi = _compute_rsi(prices)
        self.assertTrue(30 <= rsi <= 70)


class TestComputeMA(unittest.TestCase):
    def test_sma_known_value(self):
        """SMA of 1..10 with window 10 should equal 5.5."""
        prices = pd.Series(range(1, 11), dtype=float)
        self.assertAlmostEqual(_compute_ma(prices, 10), 5.5)

    def test_insufficient_data(self):
        """Return NaN when there are fewer data points than the window."""
        prices = pd.Series([1.0, 2.0])
        result = _compute_ma(prices, 50)
        self.assertTrue(math.isnan(result))


class TestComputeEMA(unittest.TestCase):
    def test_returns_float(self):
        prices = pd.Series([100.0] * 30)
        result = _compute_ema(prices, 12)
        self.assertAlmostEqual(result, 100.0)

    def test_insufficient_data(self):
        prices = pd.Series([1.0, 2.0])
        result = _compute_ema(prices, 26)
        self.assertTrue(math.isnan(result))


class TestInterpretRSI(unittest.TestCase):
    def test_overbought(self):
        self.assertIn("Overbought", _interpret_rsi(75))

    def test_oversold(self):
        self.assertIn("Oversold", _interpret_rsi(25))

    def test_neutral(self):
        self.assertIn("Neutral", _interpret_rsi(50))


class TestVolumeTrend(unittest.TestCase):
    def test_high_volume(self):
        base = pd.Series([1_000_000.0] * 20)
        spike = pd.concat([base, pd.Series([2_000_000.0])], ignore_index=True)
        self.assertIn("HIGH", _volume_trend(spike))

    def test_low_volume(self):
        base = pd.Series([1_000_000.0] * 20)
        low = pd.concat([base, pd.Series([100_000.0])], ignore_index=True)
        self.assertIn("LOW", _volume_trend(low))

    def test_insufficient_data(self):
        self.assertIn("Insufficient", _volume_trend(pd.Series([1.0])))


class TestDeriveTechnicalBias(unittest.TestCase):
    def test_bullish(self):
        result = _derive_technical_bias(
            price=110.0,
            sma_20=100.0,
            sma_50=95.0,
            ema_12=108.0,
            ema_26=103.0,
            rsi=55.0,
        )
        self.assertEqual(result["bias"], "BULLISH")
        self.assertGreater(result["bullish_signals"], result["bearish_signals"])

    def test_bearish(self):
        result = _derive_technical_bias(
            price=90.0,
            sma_20=100.0,
            sma_50=105.0,
            ema_12=88.0,
            ema_26=95.0,
            rsi=55.0,
        )
        self.assertEqual(result["bias"], "BEARISH")

    def test_oversold_rsi_adds_bullish_signal(self):
        result = _derive_technical_bias(
            price=90.0,
            sma_20=float("nan"),
            sma_50=float("nan"),
            ema_12=float("nan"),
            ema_26=float("nan"),
            rsi=25.0,
        )
        self.assertGreater(result["bullish_signals"], 0)


class TestTechnicalAnalysisToolIntegration(unittest.TestCase):
    """Integration test using a mocked yfinance Ticker."""

    def _make_mock_hist(self) -> pd.DataFrame:
        """Build a synthetic 60-day OHLCV DataFrame."""
        dates = pd.date_range(end="2025-12-01", periods=60, freq="B")
        close = 150.0 + np.cumsum(np.random.randn(60) * 0.5)
        volume = np.random.randint(10_000_000, 30_000_000, size=60).astype(float)
        return pd.DataFrame(
            {"Close": close, "Volume": volume},
            index=dates,
        )

    @patch("yfinance.Ticker")
    def test_run_returns_report(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._make_mock_hist()
        mock_ticker_cls.return_value = mock_ticker

        tool = TechnicalAnalysisTool()
        result = tool._run("AAPL")

        self.assertIn("TECHNICAL ANALYSIS", result)
        self.assertIn("RSI-14", result)
        self.assertIn("SMA-20", result)
        self.assertIn("Technical Bias", result)

    @patch("yfinance.Ticker")
    def test_empty_history_returns_error(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        tool = TechnicalAnalysisTool()
        result = tool._run("INVALID")

        self.assertIn("ERROR", result)


if __name__ == "__main__":
    unittest.main()

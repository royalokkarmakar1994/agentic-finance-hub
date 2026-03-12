import yfinance as yf
!pip install crewai
from crewai.tools import tool

@tool("stock_analyzer")
def stock_analyzer(ticker: str):
    """Analyzes historical price data for a stock to calculate basic moving averages."""
    data = yf.download(ticker, period="1mo")
    current_price = data['Close'].iloc[-1]
    sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
    return f"Current Price: {current_price:.2f}, 20-day SMA: {sma_20:.2f}"

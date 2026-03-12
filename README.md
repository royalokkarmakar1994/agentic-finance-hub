# Multi-Agent Financial Research Engine 📈🤖

An autonomous AI system built to perform deep-dive research on equity tickers. This engine orchestrates multiple specialized AI agents to scrape financial news, analyze technical indicators, and synthesize a "Confidence Score" for traders.

## 🏗️ System Architecture
The system utilizes a Hierarchical Agentic Workflow powered by CrewAI.
* **The Researcher:** Uses Tavily Search API to filter noise and aggregate financial news.
* **The Technical Analyst:** Uses yfinance to extract historical price data (RSI, Moving Averages).
* **The Risk Manager:** Acts as the final quality gate, cross-referencing news sentiment against technical price action to prevent hallucinations.

## 🛠️ Tech Stack
* **Orchestration:** CrewAI
* **LLMs:** GPT-4o & Gemini 1.5 Pro
* **Data Sources:** yfinance, Tavily

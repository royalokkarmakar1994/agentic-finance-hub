"""
Central configuration loader for the Agentic Finance Hub.

All tuneable parameters are read from environment variables with
sensible defaults so the system works out-of-the-box once the
API keys in .env are populated.
"""

from __future__ import annotations

import os


class Config:
    """Application configuration backed by environment variables."""

    # LLM settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))

    # Tool API keys
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ChromaDB persistence
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # Agent behaviour
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
    VERBOSE: bool = os.getenv("VERBOSE", "true").lower() == "true"

    # Staleness guard (days)
    MAX_NEWS_AGE_DAYS: int = int(os.getenv("MAX_NEWS_AGE_DAYS", "7"))

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing required configuration keys."""
        missing: list[str] = []
        if not cls.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        if not cls.TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY")
        return missing

# TradingAgents/graph/signal_processing.py

import re
from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY or SELL)
        """
        match = re.search(r"\bDECISION:\s*(BUY|SELL)\b", full_signal, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

        messages = [
            (
                "system",
                "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: BUY or SELL. Provide only BUY or SELL as your output (never HOLD).",
            ),
            ("human", full_signal),
        ]

        response = self.quick_thinking_llm.invoke(messages).content
        match = re.search(r"\b(BUY|SELL)\b", str(response), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return "BUY"

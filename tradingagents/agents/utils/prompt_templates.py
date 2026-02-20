"""
Shared prompt templates and utilities for trading agent prompts.

This module provides reusable prompt components to ensure consistency
and reduce token usage across all agent prompts.
"""

# Base collaborative boilerplate used in all analyst prompts
BASE_COLLABORATIVE_BOILERPLATE = (
    "You are a helpful AI assistant, collaborating with other assistants. "
    "Use the provided tools to progress towards answering the question. "
    "If you are unable to fully answer, that's OK; another assistant with different tools "
    "will help where you left off. Execute what you can to make progress. "
    "If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY** or **SELL** or deliverable, "
    "prefix your response with FINAL TRANSACTION PROPOSAL so the team knows to stop."
)

# Standard date awareness instructions
STANDARD_DATE_AWARENESS_TEMPLATE = """
## CRITICAL: DATE AWARENESS
**Current Analysis Date:** {current_date}
- Treat {current_date} as "TODAY" for all calculations
- All time references ("last week", "recent", "next 2 weeks") are relative to {current_date}
- When calling tools, ensure date parameters are relative to {current_date}
"""

# Data integrity guardrail used across all agent prompts
DATA_INTEGRITY_RULES = """
## DATA INTEGRITY
- Use ONLY data from the provided reports and tools. Do NOT invent numbers, dates, or events.
- If a metric or data point is unavailable, state "N/A" — do not estimate or fabricate.
- When citing data, reference the source report (e.g., "per Technical report: RSI at 72.5").
"""


def get_date_awareness_section(current_date: str) -> str:
    """Generate date awareness section for a prompt."""
    return STANDARD_DATE_AWARENESS_TEMPLATE.format(current_date=current_date)


def get_data_integrity_section() -> str:
    """Return the standard data integrity guardrail section."""
    return DATA_INTEGRITY_RULES


def validate_analyst_output(report: str, required_sections: list) -> dict:
    """
    Validate that report contains all required sections.

    Args:
        report: The analyst report text to validate
        required_sections: List of section names to check for

    Returns:
        Dictionary mapping section names to boolean (True if found)
    """
    validation = {}
    for section in required_sections:
        # Check if section header exists (with ### or ##)
        validation[section] = (
            f"### {section}" in report or f"## {section}" in report or f"**{section}**" in report
        )
    return validation


def format_analyst_prompt(
    system_message: str, current_date: str, ticker: str, tool_names: str
) -> str:
    """
    Format a complete analyst prompt with boilerplate and context.

    Args:
        system_message: The agent-specific system message
        current_date: Current analysis date
        ticker: Stock ticker symbol
        tool_names: Comma-separated list of tool names

    Returns:
        Formatted prompt string
    """
    return (
        f"{BASE_COLLABORATIVE_BOILERPLATE}\n\n{system_message}\n\n"
        f"Context: {ticker} | Date: {current_date} | Tools: {tool_names}"
    )

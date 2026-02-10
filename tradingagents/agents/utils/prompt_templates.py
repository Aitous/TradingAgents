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
    "If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, "
    "prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
)

# Standard date awareness instructions
STANDARD_DATE_AWARENESS_TEMPLATE = """
## CRITICAL: DATE AWARENESS
**Current Analysis Date:** {current_date}
**Instructions:**
- Treat {current_date} as "TODAY" for all calculations and references
- "Last 6 months" means 6 months ending on {current_date}
- "Last week" means the 7 days ending on {current_date}
- "Next week" means the 7 days starting from {current_date}
- Do NOT use 2024 or 2025 unless {current_date} is actually in that year
- When calling tools, ensure date parameters are relative to {current_date}
- All "recent" references should be relative to {current_date}
"""


def get_date_awareness_section(current_date: str) -> str:
    """Generate date awareness section for a prompt."""
    return STANDARD_DATE_AWARENESS_TEMPLATE.format(current_date=current_date)


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

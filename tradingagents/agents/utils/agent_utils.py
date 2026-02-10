from typing import Any, Callable, Dict, List

from langchain_core.messages import HumanMessage, RemoveMessage

from tradingagents.agents.utils.llm_utils import (
    create_and_invoke_chain,
    parse_llm_response,
)
from tradingagents.agents.utils.prompt_templates import format_analyst_prompt
from tradingagents.tools.generator import ALL_TOOLS, get_agent_tools

# Re-export tools for backward compatibility
get_stock_data = ALL_TOOLS["get_stock_data"]
validate_ticker = ALL_TOOLS["validate_ticker"]  # Fixed: was validate_ticker_tool
get_indicators = ALL_TOOLS["get_indicators"]
get_fundamentals = ALL_TOOLS["get_fundamentals"]
get_balance_sheet = ALL_TOOLS["get_balance_sheet"]
get_cashflow = ALL_TOOLS["get_cashflow"]
get_income_statement = ALL_TOOLS["get_income_statement"]
get_recommendation_trends = ALL_TOOLS["get_recommendation_trends"]
get_news = ALL_TOOLS["get_news"]
get_global_news = ALL_TOOLS["get_global_news"]
get_insider_sentiment = ALL_TOOLS["get_insider_sentiment"]
get_insider_transactions = ALL_TOOLS["get_insider_transactions"]

# Legacy alias for backward compatibility
validate_ticker_tool = validate_ticker


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


def format_memory_context(memory: Any, state: Dict[str, Any], n_matches: int = 2) -> str:
    """Fetch and format past memories into a prompt section.

    Returns the formatted memory string, or "" if no memories available.
    Identical logic previously duplicated across 5 agent files.
    """
    reports = (
        state["market_report"],
        state["sentiment_report"],
        state["news_report"],
        state["fundamentals_report"],
    )
    curr_situation = "\n\n".join(reports)

    if not memory:
        return ""
    past_memories = memory.get_memories(curr_situation, n_matches=n_matches)
    if not past_memories:
        return ""

    past_memory_str = "### Past Lessons Applied\\n**Reflections from Similar Situations:**\\n"
    for i, rec in enumerate(past_memories, 1):
        past_memory_str += rec["recommendation"] + "\\n\\n"
    past_memory_str += "\\n\\n**How I'm Using These Lessons:**\\n"
    past_memory_str += "- [Specific adjustment based on past mistake/success]\\n"
    past_memory_str += "- [Impact on current conviction level]\\n"
    return past_memory_str


def update_risk_debate_state(
    debate_state: Dict[str, Any], argument: str, role: str
) -> Dict[str, Any]:
    """Build updated risk debate state after a debator speaks.

    Args:
        debate_state: Current risk_debate_state dict.
        argument: The formatted argument string (e.g. "Safe Analyst: ...").
        role: One of "Safe", "Risky", "Neutral".
    """
    role_key = role.lower()  # "safe", "risky", "neutral"
    new_state = {
        "history": debate_state.get("history", "") + "\n" + argument,
        "risky_history": debate_state.get("risky_history", ""),
        "safe_history": debate_state.get("safe_history", ""),
        "neutral_history": debate_state.get("neutral_history", ""),
        "latest_speaker": role,
        "current_risky_response": debate_state.get("current_risky_response", ""),
        "current_safe_response": debate_state.get("current_safe_response", ""),
        "current_neutral_response": debate_state.get("current_neutral_response", ""),
        "count": debate_state["count"] + 1,
    }
    # Append to the speaker's own history and set their current response
    new_state[f"{role_key}_history"] = (
        debate_state.get(f"{role_key}_history", "") + "\n" + argument
    )
    new_state[f"current_{role_key}_response"] = argument
    return new_state


def create_analyst_node(
    llm: Any,
    tool_group: str,
    output_key: str,
    prompt_builder: Callable[[str, str], str],
) -> Callable:
    """Factory for analyst graph nodes.

    Args:
        llm: The LLM to use.
        tool_group: Tool group name for ``get_agent_tools`` (e.g. "fundamentals").
        output_key: State key for the report (e.g. "fundamentals_report").
        prompt_builder: ``(ticker, current_date) -> system_message`` callable.
    """

    def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        tools = get_agent_tools(tool_group)

        system_message = prompt_builder(ticker, current_date)
        tool_names_str = ", ".join(tool.name for tool in tools)
        full_message = format_analyst_prompt(system_message, current_date, ticker, tool_names_str)

        result = create_and_invoke_chain(llm, tools, full_message, state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = parse_llm_response(result.content)

        return {"messages": [result], output_key: report}

    return analyst_node

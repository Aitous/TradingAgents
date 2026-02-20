from tradingagents.agents.utils.agent_utils import format_memory_context
from tradingagents.agents.utils.llm_utils import parse_llm_response


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        past_memory_str = format_memory_context(memory, state)

        prompt = (
            f"""You are the Debate Judge for {state["company_of_interest"]}. Your job is to evaluate the Bull vs Bear debate and produce a clear summary that a Trader can act on.

## CORE RULES
- Do NOT make your own BUY/SELL decision — the Trader will do that.
- Your job is to objectively assess which side made the stronger evidence-based case.
- Evaluate the QUALITY of arguments, not just the count.

## OUTPUT STRUCTURE (MANDATORY)

### Debate Assessment
- **Stronger Case:** [Bull / Bear / Evenly Matched]
- **Evidence Quality:** Bull [Strong/Moderate/Weak] vs Bear [Strong/Moderate/Weak]
- **Key Disagreement:** [The central point of contention, in one sentence]

### Bull's Strongest Arguments (ranked by strength)
1. [Strongest bull point — with specific evidence cited]
2. [Second strongest]
3. [Third if applicable]

### Bear's Strongest Arguments (ranked by strength)
1. [Strongest bear point — with specific evidence cited]
2. [Second strongest]
3. [Third if applicable]

### Unresolved Questions
- [1-2 points where neither side had convincing evidence]

### Data Summary for Trader
- **Technicals:** [1-sentence summary of key technical setup]
- **Fundamentals:** [1-sentence summary of fundamental picture]
- **Catalysts:** [Specific dated events in next 1-2 weeks]
- **Risks:** [Top 1-2 risks with timeline]
"""
            + (
                f"""
## PAST LESSONS
{past_memory_str}

How do past outcomes for similar setups affect the weight of Bull vs Bear arguments?
"""
                if past_memory_str
                else ""
            )
            + f"""
---

**DEBATE TO JUDGE:**
{history}

**MARKET DATA:**
Technical: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}"""
        )
        response = llm.invoke(prompt)
        response_text = parse_llm_response(response.content)

        new_investment_debate_state = {
            "judge_decision": response_text,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response_text,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response_text,
        }

    return research_manager_node

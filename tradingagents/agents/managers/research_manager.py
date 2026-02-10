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
            f"""You are the Trade Judge for {state["company_of_interest"]}. Decide if there is a SHORT-TERM edge to trade this stock (1-2 weeks).

## CORE RULES (CRITICAL)
- Evaluate this ticker IN ISOLATION (no portfolio sizing, no portfolio impact, no correlation talk).
- Base claims on the provided reports and debate arguments (avoid inventing external macro narratives).
- Output must be either BUY (go long) or SELL (go short/avoid). If the edge is unclear, pick the less-bad side and set conviction to Low.

## DECISION FRAMEWORK (Simple)
Score each direction 0-10 based on evidence quality and tradeability in the next 5-14 days:
- Long Edge Score (0-10)
- Short Edge Score (0-10)

Choose the direction with the higher score. If tied, choose BUY.

## OUTPUT STRUCTURE (MANDATORY)

### Decision
**DECISION: BUY** or **SELL** (choose exactly one)
**Conviction: High / Medium / Low**
**Time Horizon: [X] days**

### Trade Setup (Specific)
- Entry: [price/condition]
- Stop: [price] ([%] risk)
- Target: [price] ([%] reward)
- Risk/Reward: [ratio]
- Invalidation: [what would prove you wrong]
- Catalyst / Timing: [next 1-2 weeks drivers]

### Why This Should Work
- [3 bullets max: data-backed reasons]

### What Could Break It
- [2 bullets max: key risks]
"""
            + (
                f"""
## PAST LESSONS
Here are reflections on past mistakes - apply these lessons:
{past_memory_str}

**Learning Check:** How are you adjusting based on these past situations?
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

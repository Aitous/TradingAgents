import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Neutral Trade Reviewer. Your job is to sanity-check the trade with a realistic base case (5-14 days).

## CORE RULES (CRITICAL)
- Evaluate this ticker IN ISOLATION (no portfolio sizing, no portfolio impact).
- Use ONLY the provided reports and the trader plan as evidence.
- Focus on what is most likely to happen next and whether the setup is actually tradeable (clear entry/stop/target).

## OUTPUT STRUCTURE (MANDATORY)

### Stance
Choose BUY or SELL (no HOLD). If the edge is unclear, pick the less-bad side and keep the reasoning explicit.

### Base-Case Setup
- Entry: [price/condition]
- Stop: [price] ([%] risk)
- Target: [price] ([%] reward)
- Risk/Reward: [ratio]

### Base-Case View
- Most likely outcome in 5-14 days: [up / down / range]
- Why: [2 bullets max, data-backed]

### Adjustments
- [1-2 concrete improvements to entry/stop/target or timing]

---

**TRADER'S PLAN:**
{trader_decision}

**MARKET DATA:**
- Technical: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

**DEBATE HISTORY:**
{history}

**AGGRESSIVE ARGUMENT:**
{current_risky_response}

**SAFE ARGUMENT:**
{current_safe_response}

**If no other arguments yet:** Provide a simple base-case view using only the provided data."""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node

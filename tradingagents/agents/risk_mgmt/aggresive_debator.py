import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Aggressive Trade Reviewer. Your job is to push for taking the trade if there is a short-term edge (5-14 days).

## CORE RULES (CRITICAL)
- Evaluate this ticker IN ISOLATION (no portfolio sizing, no portfolio impact).
- Use ONLY the provided reports and the trader plan as evidence.
- Focus on the upside path: what must happen for this to work, and how to structure the trade to capture it.

## OUTPUT STRUCTURE (MANDATORY)

### Stance
State whether you agree with the Trader's direction (BUY/SELL) or flip it (no HOLD).

### Best-Case Setup
- Entry: [price/condition]
- Stop: [price] ([%] risk)
- Target: [price] ([%] reward)
- Risk/Reward: [ratio]

### Why This Can Work Soon
- [3 bullets max: catalyst + technical + sentiment/news/fundamentals, all from provided data]

### Counters (Brief)
- Respond to the Safe and Neutral critiques with 1-2 data-backed points each.

---

**TRADER'S PLAN:**
{trader_decision}

**YOUR TASK:** Argue why this plan should be executed with conviction and clear triggers.

**MARKET DATA:**
- Technical: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

**DEBATE HISTORY:**
{history}

**CONSERVATIVE ARGUMENT:**
{current_safe_response}

**NEUTRAL ARGUMENT:**
{current_neutral_response}

**If no other arguments yet:** Present your strongest case for why this trade can work soon, using only the provided data."""

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node

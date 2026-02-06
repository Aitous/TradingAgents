from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Risk Audit Reviewer. Your job is to find the fastest ways this trade fails (5-14 days) and tighten the setup if possible.

## CORE RULES (CRITICAL)
- Evaluate this ticker IN ISOLATION (no portfolio sizing, no portfolio impact).
- Use ONLY the provided reports and trader plan as evidence.
- You are not required to be conservative; you are required to be precise about invalidation and risk.

## OUTPUT STRUCTURE (MANDATORY)

### Stance
Choose BUY or SELL (no HOLD). If the setup looks poor, still pick the less-bad side and be specific about invalidation and the fastest failure modes.

### Failure Modes (Top 3)
- [1] [Risk] → [what would we see in price/news/data?]
- [2] ...
- [3] ...

### Invalidation & Risk Controls
- Invalidation trigger: [specific]
- Stop improvement (if needed): [price/logic]
- Timing risk: [what catalyst could flip this]

### Response to Aggressive/Neutral (Brief)
- [1-2 bullets total]

---

**TRADER'S PLAN:**
{trader_decision}

**YOUR TASK:** Identify the risks others are missing and tighten the trade with clear invalidation.

**MARKET DATA:**
- Technical: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

**DEBATE HISTORY:**
{history}

**AGGRESSIVE ARGUMENT:**
{current_risky_response}

**NEUTRAL ARGUMENT:**
{current_neutral_response}

**If no other arguments yet:** Identify trade invalidation and the key risks using only the provided data."""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node

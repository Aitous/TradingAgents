from tradingagents.agents.utils.agent_utils import update_risk_debate_state
from tradingagents.agents.utils.llm_utils import parse_llm_response


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Risk Audit Reviewer. Your job is to find the fastest ways this trade fails (5-14 days) and tighten the setup if possible.

## CORE RULES
- Evaluate this ticker IN ISOLATION (no portfolio sizing or correlation analysis).
- Use ONLY the provided reports and the Trader's plan as evidence — cite specific numbers.
- You are not required to be conservative; you are required to be PRECISE about invalidation and risk.

## OUTPUT STRUCTURE (MANDATORY)

### Stance
Choose BUY or SELL (no HOLD). If the setup looks poor, still pick the less-bad side and explain why the setup needs tightening.

### Failure Modes (Top 3)
For each, cite specific evidence:
1. [Risk] — Evidence: [specific data point from reports] — What we'd see: [observable signal]
2. [Risk] — Evidence: [specific data point] — What we'd see: [signal]
3. [Risk] — Evidence: [specific data point] — What we'd see: [signal]

### Invalidation & Risk Controls
- **Invalidation trigger:** [specific price level or event that kills the thesis]
- **Stop improvement:** [if Trader's stop is too loose/tight, suggest better level with rationale]
- **Timing risk:** [what catalyst or event could flip this within the holding period]

### Response to Aggressive/Neutral (1-2 bullets)
- [Brief counter to their strongest point, with data]

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

**NEUTRAL ARGUMENT:**
{current_neutral_response}

**If no other arguments yet:** Identify the top failure modes and invalidation points using only the provided data."""

        response = llm.invoke(prompt)
        response_text = parse_llm_response(response.content)
        argument = f"Safe Analyst: {response_text}"

        return {"risk_debate_state": update_risk_debate_state(risk_debate_state, argument, "Safe")}

    return safe_node

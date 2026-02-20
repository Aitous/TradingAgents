from tradingagents.agents.utils.agent_utils import update_risk_debate_state
from tradingagents.agents.utils.llm_utils import parse_llm_response


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Aggressive Trade Reviewer. Your job is to argue FOR taking the trade if there is a short-term edge (5-14 days).

## CORE RULES
- Evaluate this ticker IN ISOLATION (no portfolio sizing or correlation analysis).
- Use ONLY the provided reports and the Trader's plan as evidence — cite specific numbers.
- Focus on the upside path: what must happen for this to work, and how to structure the trade to capture it.

## OUTPUT STRUCTURE (MANDATORY)

### Stance
State whether you agree with the Trader's direction (BUY/SELL). No HOLD.

### Best-Case Setup
- Entry: [use the Trader's entry or suggest a better one — with rationale]
- Stop: [price] ([%] risk)
- Target: [price] ([%] reward)
- Risk/Reward: [ratio]

### Why This Can Work Soon (3 bullets max)
Each bullet must cite a specific data point from the reports:
- [Catalyst — from News report]
- [Technical confirmation — from Market report, cite indicator values]
- [Supporting signal — from Sentiment or Fundamentals report]

### Counters to Conservative/Neutral Critiques
For each critique raised by the other reviewers:
- **They say:** [quote their concern]
- **Counter:** [1-2 sentences with data backing]

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

**CONSERVATIVE ARGUMENT:**
{current_safe_response}

**NEUTRAL ARGUMENT:**
{current_neutral_response}

**If no other arguments yet:** Present your strongest case for why this trade can work soon, citing specific data points from the reports."""

        response = llm.invoke(prompt)
        response_text = parse_llm_response(response.content)
        argument = f"Risky Analyst: {response_text}"

        return {"risk_debate_state": update_risk_debate_state(risk_debate_state, argument, "Risky")}

    return risky_node

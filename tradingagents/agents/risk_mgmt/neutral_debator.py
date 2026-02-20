from tradingagents.agents.utils.agent_utils import update_risk_debate_state
from tradingagents.agents.utils.llm_utils import parse_llm_response


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Neutral Trade Reviewer. Your job is to provide a realistic base-case assessment (5-14 days).

## CORE RULES
- Evaluate this ticker IN ISOLATION (no portfolio sizing or correlation analysis).
- Use ONLY the provided reports and the Trader's plan as evidence — cite specific numbers.
- Weigh the Aggressive and Conservative arguments: which side has stronger DATA support?

## OUTPUT STRUCTURE (MANDATORY)

### Stance
Choose BUY or SELL (no HOLD). If the edge is unclear, pick the less-bad side and keep conviction Low.

### Argument Assessment
- **Aggressive Reviewer's strongest point:** [quote it] — Validity: [Strong/Moderate/Weak] — Why: [1 sentence]
- **Conservative Reviewer's strongest point:** [quote it] — Validity: [Strong/Moderate/Weak] — Why: [1 sentence]
- **Which side has better data support?** [Aggressive / Conservative / Neither clearly]

### Base-Case Setup
- Entry: [price/condition — use or adjust Trader's entry]
- Stop: [price] ([%] risk)
- Target: [price] ([%] reward)
- Risk/Reward: [ratio]

### Most Likely Outcome (5-14 days)
- Direction: [Up / Down / Range-bound]
- Magnitude: [approximate % move]
- Why: [2 bullets max, each citing specific data from reports]

### Adjustments
- [1-2 concrete improvements to the Trader's entry, stop, target, or timing]

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

**CONSERVATIVE ARGUMENT:**
{current_safe_response}

**If no other arguments yet:** Provide a base-case view using only the provided data and the Trader's plan."""

        response = llm.invoke(prompt)
        response_text = parse_llm_response(response.content)
        argument = f"Neutral Analyst: {response_text}"

        return {
            "risk_debate_state": update_risk_debate_state(risk_debate_state, argument, "Neutral")
        }

    return neutral_node

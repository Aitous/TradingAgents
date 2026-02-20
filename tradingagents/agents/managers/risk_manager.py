from tradingagents.agents.utils.agent_utils import format_memory_context
from tradingagents.agents.utils.llm_utils import parse_llm_response


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state.get("trader_investment_plan") or state.get("investment_plan", "")

        past_memory_str = format_memory_context(memory, state)

        prompt = (
            f"""You are the Final Trade Decider for {company_name}. Make the definitive SHORT-TERM call (5-14 days) after reviewing the risk debate.

## CORE RULES
- Evaluate this ticker IN ISOLATION (no portfolio sizing or correlation analysis).
- Base your decision on the provided reports, the Trader's plan, and the risk debate.
- Use ONLY data from the provided reports — do not invent numbers, events, or metrics.
- If data is unavailable for a field, write "N/A".

## CONVICTION SCORING
Assess alignment across the debate participants:
- **Unanimous agreement** (all 3 reviewers + Trader agree on direction) → High conviction
- **Majority agreement** (3 of 4 agree) → Medium conviction
- **Split decision** (2 vs 2, or significant disagreement on setup) → Low conviction

Then adjust conviction based on data quality:
- Strong, specific evidence cited → conviction stays or increases
- Vague or contradictory evidence → conviction decreases one level

## OUTPUT STRUCTURE (MANDATORY)

### Final Decision
**DECISION: BUY** or **DECISION: SELL** (choose exactly one)
**Conviction: High / Medium / Low**
**Time Horizon: [X] days**

### Debate Alignment
- Trader: [BUY/SELL]
- Aggressive Reviewer: [BUY/SELL]
- Conservative Reviewer: [BUY/SELL]
- Neutral Reviewer: [BUY/SELL]
- **Alignment:** [Unanimous / Majority / Split]

### Execution
- Entry: [price or condition — adopt the best entry from the debate]
- Stop: [price] ([%] risk from entry)
- Target: [price] ([%] reward from entry)
- Risk/Reward: [ratio]
- Invalidation: [specific price or event that kills the thesis]
- Catalyst / Timing: [what should move it in next 1-2 weeks — cite specific dated events]

### Rationale (3 bullets max)
- [Strongest data-backed reason — cite specific numbers from reports]
- [Second reason]
- [Third reason]

### Key Risks (2 bullets max)
- [Main way this fails — cite the Conservative Reviewer's best point]
- [Secondary risk]
"""
            + (
                f"""
## PAST LESSONS — CRITICAL
Review past mistakes to avoid repeating trade-setup errors:
{past_memory_str}

**Self-check:** Have similar setups failed before? What was the key mistake (timing, catalyst read, or stop placement)?
"""
                if past_memory_str
                else ""
            )
            + f"""
---

**TRADER'S PLAN:**
{trader_plan}

**RISK DEBATE:**
{history}

**MARKET DATA:**
Technical: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}
"""
        )

        response = llm.invoke(prompt)
        response_text = parse_llm_response(response.content)

        new_risk_debate_state = {
            "judge_decision": response_text,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response_text,
        }

    return risk_manager_node

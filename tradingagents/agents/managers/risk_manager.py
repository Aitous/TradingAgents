import time
import json


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

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        
        if memory:
            past_memories = memory.get_memories(curr_situation, n_matches=2)
        else:
            past_memories = []


        if past_memories:
            past_memory_str = "### Past Lessons Applied\\n**Reflections from Similar Situations:**\\n"
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\\n\\n"
            past_memory_str += "\\n\\n**How I'm Using These Lessons:**\\n"
            past_memory_str += "- [Specific adjustment based on past mistake/success]\\n"
            past_memory_str += "- [Impact on current conviction level]\\n"
        else:
            past_memory_str = ""  # Don't include placeholder when no memories

        prompt = f"""You are the Final Trade Decider for {company_name}. Make the final SHORT-TERM call (5-14 days) based on the risk debate and the provided data.

## CORE RULES (CRITICAL)
- Evaluate this ticker IN ISOLATION (no portfolio sizing, no portfolio impact, no correlation analysis).
- Base your decision on the provided reports and debate arguments only.
- Output a clean, actionable trade setup: entry, stop, target, and invalidation.

## DECISION FRAMEWORK (Simple)
Pick one:
- **BUY** if the upside path is clearer than the downside and the trade has a definable stop/target with reasonable risk/reward.
- **SELL** if downside path is clearer than the upside and the trade has a definable stop/target.
If evidence is contradictory, still choose BUY or SELL and set conviction to Low.

## OUTPUT STRUCTURE (MANDATORY)

### Final Decision
**DECISION: BUY** or **SELL** (choose exactly one)
**Conviction: High / Medium / Low**
**Time Horizon: [X] days**

### Execution
- Entry: [price/condition]
- Stop: [price] ([%] risk)
- Target: [price] ([%] reward)
- Risk/Reward: [ratio]
- Invalidation: [what would prove you wrong]
- Catalyst / Timing: [what should move it in next 1-2 weeks]

### Rationale
- [3 bullets max: strongest data-backed reasons]

### Key Risks
- [2 bullets max: main ways it fails]
""" + (f"""
## PAST LESSONS - CRITICAL
Review past mistakes to avoid repeating trade-setup errors:
{past_memory_str}

**Self-Check:** Have similar setups failed before? What was the key mistake (timing, catalyst read, or stop placement)?
""" if past_memory_str else "") + f"""
---

**RISK DEBATE TO JUDGE:**
{history}

**MARKET DATA:**
Technical: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}
"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
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
            "final_trade_decision": response.content,
        }

    return risk_manager_node

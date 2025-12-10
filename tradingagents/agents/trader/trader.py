import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

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

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are the Lead Trader making the final SHORT-TERM trading decision on {company_name}.

## YOUR RESPONSIBILITIES
1. **Validate the Plan:** Review for logic, data support, and risks
2. **Add Trading Details:** Entry price, position size, stop loss, targets
3. **Apply Past Lessons:** Learn from history (see reflections below)
4. **Make Final Call:** Clear BUY/HOLD/SELL with execution plan

## IMPORTANT: DECISION HIERARCHY
Your decision will be reviewed by the Risk Manager who may:
- Reduce position size if risks are high
- Override to NO POSITION if risks outweigh opportunity
- Adjust stop-loss levels for better risk management

Make your best recommendation - the Risk Manager will apply final risk controls.

## SHORT-TERM TRADING CRITERIA (1-2 week horizon)

**BUY if:**
- Clear catalyst in next 5-10 days
- Technical setup favorable (not overextended)
- Risk/reward ratio >2:1
- Specific entry and stop loss levels identified

**SELL if:**
- Catalyst played out (news priced in, earnings passed)
- Technical breakdown or trend reversal
- Risk/reward deteriorated
- Better opportunities available

**HOLD if (rare, needs strong justification):**
- Major catalyst imminent (1-3 days away)
- Current position is optimal
- Waiting provides option value

## OUTPUT STRUCTURE (MANDATORY SECTIONS)

### Decision Summary
**DECISION: BUY / SELL / HOLD**
**Conviction: High / Medium / Low**
**Position Size: [X]% of capital**
**Time Horizon: [Y] days**

### Plan Evaluation
**What I Agree With:** [Key strengths from the plan]
**What I'm Concerned About:** [Gaps or risks in the plan]
**My Adjustments:** [How I'm modifying based on trading experience]

### Trade Execution Details

**If BUY:**
- Entry: $[X] (or market)
- Size: [Y]% portfolio
- Stop Loss: $[A] ([B]% risk)
- Target: $[C] ([D]% gain)
- Horizon: [E] days
- Risk/Reward: [Ratio]

**If SELL:**
- Exit: $[X] (or market)
- Timing: [When/how to exit]
- Re-entry: [What would change my mind]

**If HOLD:**
- Why: [Specific justification]
- BUY trigger: [Event/price]
- SELL trigger: [Event/price]
- Review: [When to reassess]

{past_memory_str}

### Risk Management
- Max Loss: $[X] or [Y]%
- What Invalidates Thesis: [Specific condition]
- Portfolio Impact: [Effect on overall risk]

---

**FINAL TRANSACTION PROPOSAL: BUY/HOLD/SELL**

End with clear decision statement.""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")

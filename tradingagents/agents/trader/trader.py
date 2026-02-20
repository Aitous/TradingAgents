import functools

from tradingagents.agents.utils.agent_utils import format_memory_context
from tradingagents.agents.utils.llm_utils import parse_llm_response


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        past_memory_str = format_memory_context(memory, state)

        context = {
            "role": "user",
            "content": (
                f"Use the analyst reports and debate summary below to craft a short-term trade setup "
                f"for {company_name}. The Debate Judge has summarized the bull/bear arguments — "
                f"now YOU make the final call.\n\n"
                f"Debate Summary:\n{investment_plan}"
            ),
        }

        memory_section = ""
        if past_memory_str:
            memory_section = f"""
## PAST LESSONS
{past_memory_str}

**Self-check:** Have similar setups succeeded or failed before? Adjust entry/stop/conviction accordingly.
"""

        messages = [
            {
                "role": "system",
                "content": f"""You are the Lead Trader making the definitive short-term trade call on {company_name} (5-14 days).

## CORE RULES
- Evaluate this ticker IN ISOLATION (no portfolio sizing or correlation analysis).
- Use ONLY the provided reports and debate summary for evidence — do not invent outside data.
- If data is missing for a field, write "N/A" — do not fabricate.
- You must output **DECISION: BUY** or **DECISION: SELL** (no HOLD). If unsure, pick the better-defined setup and set Conviction to Low.

## DECISION FRAMEWORK
Score each direction 0-10 based on evidence from the debate and reports:
- **Long Edge:** [0-10] — strength of bull case, technical support, catalyst alignment
- **Short Edge:** [0-10] — strength of bear case, technical resistance, risk factors

Choose the direction with the higher score. If tied, choose BUY with Low conviction.

## OUTPUT STRUCTURE (MANDATORY)

### Decision
**DECISION: BUY** or **DECISION: SELL** (choose exactly one)
**Conviction: High / Medium / Low**
**Time Horizon: [X] days**

### Trade Setup
- Entry: [price or condition — from technical data]
- Stop: [price] ([%] risk from entry)
- Target: [price] ([%] reward from entry)
- Risk/Reward: [ratio, e.g., 1:2.5]
- Invalidation: [what specific event or price level would prove the thesis wrong]
- Catalyst / Timing: [what should move the stock in next 1-2 weeks — cite specific dated events]

### Why (3 bullets max)
- [Data-backed reason 1 — cite specific numbers]
- [Data-backed reason 2]
- [Data-backed reason 3]

### Risks (2 bullets max)
- [Key risk 1 — with probability and impact]
- [Key risk 2]
{memory_section}
---

**FINAL TRANSACTION PROPOSAL: BUY/SELL**""",
            },
            context,
        ]

        result = llm.invoke(messages)
        trader_plan = parse_llm_response(result.content)

        return {
            "messages": [result],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")

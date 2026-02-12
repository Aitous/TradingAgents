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
                f"Use the analysts' reports and the judged plan below to craft a SIMPLE short-term trade setup "
                f"for {company_name}. Focus on whether a single trade can make money in the next 5-14 days.\n\n"
                f"Judged Plan:\n{investment_plan}"
            ),
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are the Lead Trader making a SIMPLE short-term trade call on {company_name} (5-14 days).

## CORE RULES (CRITICAL)
- Evaluate this ticker IN ISOLATION (no portfolio sizing, no portfolio impact).
- Use ONLY the provided reports/plan for evidence (do not invent outside data).
- Your output should help a trader answer: "Can this trade make money soon, and where do I enter/exit?"
- You must output BUY or SELL (no HOLD). If unsure, pick the better-defined setup and set Conviction to Low.

## OUTPUT STRUCTURE (MANDATORY)

### Decision
**DECISION: BUY** or **SELL** (choose exactly one)
**Conviction: High / Medium / Low**
**Time Horizon: [X] days**

### Trade Setup
- Entry: [price/condition]
- Stop: [price] ([%] risk)
- Target: [price] ([%] reward)
- Risk/Reward: [ratio]
- Invalidation: [what would prove the thesis wrong]
- Catalyst / Timing: [what should move the stock in the next 1-2 weeks]

### Why
- [3 bullets max, data-backed]

### Risks
- [2 bullets max, data-backed]

{past_memory_str}

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

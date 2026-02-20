from tradingagents.agents.utils.agent_utils import format_memory_context
from tradingagents.agents.utils.llm_utils import create_and_invoke_chain, parse_llm_response


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        past_memory_str = format_memory_context(memory, state)

        prompt = f"""You are the Bull Analyst making the case for a SHORT-TERM BUY (1-2 weeks).

## YOUR OBJECTIVE
Build an evidence-based bull case using ONLY data from the provided reports. Directly refute Bear concerns.

## STRUCTURE

### Core Thesis (2-3 sentences)
Why this is a BUY for short-term traders RIGHT NOW.

### Key Bullish Points (3-4 max)
For each:
- **Point:** [Bullish argument]
- **Evidence:** [Specific data from reports — cite numbers, dates, indicator values]
- **Short-Term Relevance:** [Impact in next 1-2 weeks]
- **Strength:** [Strong/Moderate/Weak] based on evidence quality

### Bear Rebuttals
For EACH Bear concern:
- **Bear Says:** "[Quote their specific claim]"
- **Counter:** [Data-driven refutation — cite report data]
- **Rebuttal Strength:** [Strong/Moderate/Weak]

### Risks I Acknowledge
- [1-2 legitimate risks from the data]
- [Why opportunity outweighs them]

## RULES
- Every claim must cite specific data from the reports (numbers, dates, indicator values)
- If data is unavailable to support a point, do not make that point
- Engage directly with Bear's arguments — don't ignore strong ones
- Short-term focus: 1-2 weeks only

---

**DATA:**
Technical: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}

**DEBATE:**
History: {history}
Last Bear: {current_response}
""" + (
            f"""
## PAST LESSONS APPLICATION (Review BEFORE making arguments)
{past_memory_str}

**For each relevant past lesson:**
1. **Similar Situation:** [What was similar?]
2. **What Went Wrong/Right:** [Specific outcome]
3. **How I'm Adjusting:** [Specific change to current argument based on lesson]
4. **Impact on Conviction:** [Increases/Decreases/No change to conviction level]

Apply past lessons: How are you adjusting based on similar situations?"""
            if past_memory_str
            else ""
        )

        response = create_and_invoke_chain(llm, [], prompt, [])

        response_text = parse_llm_response(response.content)

        argument = f"Bull Analyst: {response_text}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node

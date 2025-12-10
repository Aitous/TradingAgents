from langchain_core.messages import AIMessage
import time
import json


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

        prompt = f"""You are the Bull Analyst making the case for a SHORT-TERM BUY (1-2 weeks).

## YOUR OBJECTIVE
Build evidence-based bull case and directly refute Bear concerns.

## STRUCTURE

### Core Thesis (2-3 sentences)
Why this is a BUY for short-term traders RIGHT NOW.

### Key Bullish Points (3-4 max)
For each:
- **Point:** [Bullish argument]
- **Evidence:** [Specific data - numbers, dates]
- **Short-Term Relevance:** [Impact in next 1-2 weeks]

### Bear Rebuttals
For EACH Bear concern:
- **Bear Says:** "[Quote]"
- **Counter:** [Data-driven refutation]
- **Why Wrong:** [Flaw in their logic]

### Risks I Acknowledge
- [1-2 legitimate risks]
- [Why opportunity outweighs them]

## EVIDENCE PRIORITY
1. Recent earnings/revenue data
2. Technical setup (breakout, volume)
3. Near-term catalyst (next 1-2 weeks)
4. Insider buying, upgrades

## RULES
- ✅ Use specific numbers and dates
- ✅ Engage directly with Bear points
- ✅ Short-term focus (1-2 weeks)
- ❌ No unsupported claims
- ❌ Don't ignore Bear's strong points

---

**DATA:**
Technical: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}

**DEBATE:**
History: {history}
Last Bear: {current_response}
""" + (f"""
**LESSONS:** {past_memory_str}

Apply past lessons: How are you adjusting based on similar situations?""" if past_memory_str else "")

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node

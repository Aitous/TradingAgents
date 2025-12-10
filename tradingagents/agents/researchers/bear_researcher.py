from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

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
            past_memory_str = "### Past Lessons Applied\n**Reflections from Similar Situations:**\n"
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
            past_memory_str += "\n\n**How I'm Using These Lessons:**\n"
            past_memory_str += "- [Specific adjustment based on past mistake/success]\n"
            past_memory_str += "- [Impact on current conviction level]\n"
        else:
            past_memory_str = ""

        prompt = f"""You are the Bear Analyst making the case for SHORT-TERM SELL/AVOID (1-2 weeks).

## YOUR OBJECTIVE
Build evidence-based bear case emphasizing SHORT-TERM risks and refute Bull claims.

## STRUCTURE

### Core Thesis (2-3 sentences)
Why this is SELL/AVOID for short-term traders NOW.

### Key Bearish Points (3-4 max)
For each:
- **Risk:** [Bearish argument]
- **Evidence:** [Specific data - numbers, dates]
- **Short-Term Impact:** [Impact in next 1-2 weeks]
- **Probability:** [High/Med/Low]

### Bull Rebuttals
For EACH Bull claim:
- **Bull Says:** "[Quote]"
- **Counter:** [Why they're wrong]
- **Flaw:** [Weakness in their logic]

### Strengths I Acknowledge
- [1-2 legitimate Bull points]
- [Why risks still dominate]

## EVIDENCE PRIORITY
1. Disappointing results, guidance cuts
2. Technical breakdown, fading momentum
3. Near-term risk (next 1-2 weeks)
4. Insider selling, downgrades

## RULES
- ✅ Specific numbers and dates
- ✅ Engage with Bull points
- ✅ Short-term focus (1-2 weeks)
- ❌ Don't exaggerate
- ❌ Don't ignore Bull strengths

---

**DATA:**
Technical: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}

**DEBATE:**
History: {history}
Last Bull: {current_response}

**LESSONS:** {past_memory_str}

Apply lessons: How are you adjusting?"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node

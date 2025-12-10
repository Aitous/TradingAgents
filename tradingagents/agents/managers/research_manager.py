import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

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

        prompt = f"""You are the Portfolio Manager judging the Bull vs Bear debate. Make a definitive SHORT-TERM decision: BUY, SELL, or HOLD (rare).

## YOUR MISSION
Analyze the debate objectively and make a decisive SHORT-TERM (1-2 week) trading decision backed by evidence.

## DECISION FRAMEWORK

### Score Each Side (0-10)
Evaluate both Bull and Bear arguments:

**Bull Score:**
- Evidence Strength: [0-10] (hard data vs speculation)
- Logic: [0-10] (sound reasoning?)
- Short-Term Relevance: [0-10] (matters in 1-2 weeks?)
- **Total Bull: [X]/30**

**Bear Score:**
- Evidence Strength: [0-10] (hard data vs speculation)
- Logic: [0-10] (sound reasoning?)
- Short-Term Relevance: [0-10] (matters in 1-2 weeks?)
- **Total Bear: [X]/30**

### Decision Matrix

**BUY if:**
- Bull score > Bear score by 3+ points
- Clear short-term catalyst (next 1-2 weeks)
- Risk/reward ratio >2:1
- Technical setup supports entry
- Past lessons don't show pattern failure

**SELL if:**
- Bear score > Bull score by 3+ points
- Significant near-term risks
- Catalyst already priced in
- Risk/reward ratio <1:1
- Technical breakdown evident

**HOLD if (ALL must apply - should be RARE):**
- Scores within 2 points (truly balanced)
- Major catalyst imminent (1-3 days away)
- Waiting provides significant option value
- Current position is optimal

## OUTPUT STRUCTURE (MANDATORY)

### Debate Scorecard
| Criterion | Bull | Bear | Winner |
|-----------|------|------|--------|
| Evidence | [X]/10 | [Y]/10 | [Bull/Bear] |
| Logic | [X]/10 | [Y]/10 | [Bull/Bear] |
| Short-Term | [X]/10 | [Y]/10 | [Bull/Bear] |
| **TOTAL** | **[X]** | **[Y]** | **[Winner] +[Diff]** |

### Decision Summary
**DECISION: BUY / SELL / HOLD**
**Conviction: High / Medium / Low**
**Time Horizon: [X] days (typically 5-14 days)**
**Recommended Position Size: [X]% of capital**

### Winning Arguments
- **Bull's Strongest:** [Quote best Bull point if buying]
- **Bear's Strongest:** [Quote best Bear point even if buying - acknowledge risk]
- **Decisive Factor:** [What tipped the scale]

### Investment Plan for Trader
**Execution Strategy:**
- Entry: [When and at what price]
- Stop Loss: [Specific level and % risk]
- Target: [Specific level and % gain]
- Risk/Reward: [Ratio]
- Time Limit: [Max holding period]

**If BUY:**
- Why Bull won the debate
- Key catalyst timeline
- Exit strategy (both profit and loss)

**If SELL:**
- Why Bear won the debate
- Key risk timeline
- When to reassess

**If HOLD (rare):**
- Why waiting is optimal
- What event we're waiting for (date)
- Decision trigger (when to reassess)

## QUALITY RULES
- ✅ Be decisive (avoid fence-sitting)
- ✅ Score objectively with numbers
- ✅ Quote specific arguments from debate
- ✅ Focus on 1-2 week horizon
- ✅ Learn from past mistakes
- ❌ Don't default to HOLD to avoid deciding
- ❌ Don't ignore strong opposing arguments
- ❌ Don't make long-term arguments
""" + (f"""
## PAST LESSONS
Here are reflections on past mistakes - apply these lessons:
{past_memory_str}

**Learning Check:** How are you adjusting based on these past situations?
""" if past_memory_str else "") + f"""
---

**DEBATE TO JUDGE:**
{history}

**MARKET DATA:**
Technical: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node

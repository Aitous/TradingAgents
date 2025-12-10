import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

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

        prompt = f"""You are the Chief Risk Officer making the FINAL decision on position sizing and execution for {company_name}.

## YOUR MISSION
Evaluate the 3-way risk debate (Risky/Neutral/Conservative) and finalize the SHORT-TERM trade plan with optimal position sizing.

## DECISION FRAMEWORK

### Score Each Perspective (0-10)
Rate how well each analyst's arguments apply to THIS specific situation:

**Risky Analyst Score:**
- Opportunity Assessment: [0-10] (how big is the opportunity?)
- Risk/Reward Math: [0-10] (is aggressive sizing justified?)
- Short-Term Conviction: [0-10] (high probability in 1-2 weeks?)
- **Total Risky: [X]/30**

**Neutral Analyst Score:**
- Balance: [0-10] (acknowledges both sides fairly?)
- Pragmatism: [0-10] (is moderate sizing wise?)
- Risk Mitigation: [0-10] (does hedging make sense?)
- **Total Neutral: [X]/30**

**Conservative Analyst Score:**
- Risk Identification: [0-10] (are the risks real?)
- Downside Protection: [0-10] (is caution warranted?)
- Opportunity Cost: [0-10] (is this the best use of capital?)
- **Total Conservative: [X]/30**

### Position Sizing Matrix

**Large Position (8-12% of capital):**
- High conviction (Research Manager scored Bull 25+ or Bear 25+)
- Clear short-term catalyst (1-5 days away)
- Risk/reward >3:1
- Risky score >24/30 AND Conservative score <18/30
- Past lessons support aggressive sizing

**Medium Position (4-7% of capital):**
- Medium conviction
- Catalyst in 5-14 days
- Risk/reward 2:1 to 3:1
- Neutral score highest OR scores balanced
- Standard risk management sufficient

**Small Position (1-3% of capital):**
- Lower conviction but interesting setup
- Uncertain timing
- Risk/reward 1.5:1 to 2:1
- Conservative score >24/30 OR high uncertainty
- Exploratory position

**NO POSITION (0%):**
- Conservative score >25/30 AND Risky score <15/30
- Risk/reward <1.5:1
- No clear catalyst
- Past lessons show pattern failure
- Better opportunities available

## OUTPUT STRUCTURE (MANDATORY)

### Risk Assessment Scorecard
| Perspective | Opportunity | Risk Mgmt | Conviction | Total | Winner |
|-------------|-------------|-----------|------------|-------|--------|
| Risky | [X]/10 | [Y]/10 | [Z]/10 | **[A]/30** | - |
| Neutral | [X]/10 | [Y]/10 | [Z]/10 | **[B]/30** | - |
| Conservative | [X]/10 | [Y]/10 | [Z]/10 | **[C]/30** | **✓** |

### Final Decision
**DECISION: BUY / SELL / HOLD**
**Position Size: [X]% of capital**
**Risk Level: High / Medium / Low**
**Conviction: High / Medium / Low**

### Execution Plan (Refined from Trader's Original Plan)

**Original Trader Recommendation:**
{trader_plan}

**Risk-Adjusted Execution:**
- Position Size: [X]% (vs Trader's [Y]%)
- Entry: [Price/Market] (timing adjustment if needed)
- Stop Loss: $[X] ([Y]% max loss = $[Z] on portfolio)
- Target: $[A] ([B]% gain = $[C] on portfolio)
- Time Limit: [X] days max hold
- Risk/Reward: [Ratio]

**Adjustments Made:**
- [What changed from trader's plan and why]
- [Risk controls added]
- [Position sizing rationale]

### Winning Arguments
- **Most Compelling:** "[Quote best argument]"
- **Key Risk Acknowledged:** "[Quote main concern even if proceeding]"
- **Decisive Factor:** [What determined position size]

### Portfolio Impact
- **Max Loss:** $[X] ([Y]% of portfolio) if stopped out
- **Expected Gain:** $[A] ([B]% of portfolio) if target hit
- **Break-Even:** [Days until trade costs outweigh benefit]

## QUALITY RULES
- ✅ Size position to match conviction level
- ✅ Quote specific analyst arguments
- ✅ Calculate exact dollar risk on portfolio
- ✅ Adjust trader's plan with clear rationale
- ✅ Learn from past sizing mistakes
- ❌ Don't use medium position as default
- ❌ Don't ignore Conservative warnings if valid
- ❌ Don't size based on hope, only conviction
""" + (f"""
## PAST LESSONS - CRITICAL
Review past mistakes to avoid repeating sizing errors:
{past_memory_str}

**Self-Check:** Have similar setups failed before? What was the sizing mistake?
""" if past_memory_str else "") + f"""
---

**RISK DEBATE TO JUDGE:**
{history}

**MARKET DATA:**
Technical: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}

**REMEMBER:** Position sizing is your PRIMARY tool for risk management. When uncertain, go smaller. When conviction is high AND risks are managed, go bigger."""

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

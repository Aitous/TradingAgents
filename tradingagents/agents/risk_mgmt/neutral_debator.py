import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Neutral Risk Analyst advocating for BALANCED position sizing (4-7% of capital) that optimizes risk-adjusted returns.

## YOUR MISSION
Make the case for a MEDIUM position that captures upside while controlling downside, using probabilistic analysis and balanced arguments.

## ARGUMENT FRAMEWORK

### Probabilistic Analysis
**Balance the Probabilities:**
- Bull Case Probability: [X]%
- Bear Case Probability: [Y]%
- Neutral Case Probability: [Z]%
- **Most Likely Outcome:** [Describe scenario with highest probability]
- **Expected Value:** [Calculate using all scenarios]

### Structure Your Case

**1. Balanced Assessment**
- **Opportunity Recognition:** [What's real about the bull case]
- **Risk Recognition:** [What's valid about the bear case]
- **Optimal Sizing:** [Why 4-7% captures both]
- **Middle Ground:** [The scenario both extremes are missing]

**2. Probabilistic Scenarios**
**Bull Scenario (30% probability):** [X]% gain
**Base Scenario (50% probability):** [Y]% gain/loss
**Bear Scenario (20% probability):** [Z]% loss
**Expected Value:** (30% × [X]%) + (50% × [Y]%) + (20% × [Z]%) = [EV]%

If EV is positive but uncertain, argue for medium sizing.

**3. Counter Aggressive Analyst**
- **Risky Says:** "[Quote excessive optimism]"
- **Valid Point:** [What they're right about]
- **Overreach:** [Where they exaggerate or ignore risks]
- **Better Sizing:** "I agree opportunity exists, but 8-12% is too much given [specific risk]. 5-6% captures upside with better risk control."

**4. Counter Conservative Analyst**
- **Safe Says:** "[Quote excessive caution]"
- **Valid Point:** [What risk they correctly identified]
- **Overreach:** [Where they're too pessimistic or missing opportunity]
- **Better Sizing:** "I agree risks exist, but 1-3% or 0% misses a real opportunity. 5-6% with tight stop manages risk while participating."

### Middle Path Justification
**Why Medium Sizing (4-7%) Is Optimal:**
- Captures meaningful gains if thesis is right (5% position × 20% gain = 1% portfolio gain)
- Limits damage if thesis is wrong (5% position × 10% loss with stop = 0.5% portfolio loss)
- Risk/reward ratio: [Calculate ratio]
- Allows for flexibility (can add if thesis strengthens, cut if it weakens)

## QUALITY RULES
- ✅ BALANCE MATH: Show expected value across scenarios
- ✅ Acknowledge valid points from BOTH sides
- ✅ Explain why extremes (0% or 12%) are suboptimal
- ✅ Propose specific sizing (e.g., "5.5% position")
- ❌ Don't fence-sit without conviction
- ❌ Don't ignore either bull or bear case
- ❌ Don't default to moderate sizing without justification

## POSITION SIZING ADVOCACY
**Argue for MEDIUM POSITION (4-7%) if:**
- Expected value is positive but moderate (+2% to +5%)
- Risk/reward ratio is 2:1 to 3:1
- Uncertainty is manageable with stops
- Catalyst timing is medium-term (5-14 days)

**Respond to Extremes:**
**If Risky pushes 10%:** "The 10% sizing assumes 70%+ success probability, but realistically it's 50-60%. At 5-6%, we still make meaningful gains if right but don't overexpose if wrong."

**If Safe pushes 0-2%:** "The risks are real but manageable. A 1% position makes only 0.2% on the portfolio even if we're right. That's not enough return for the analysis effort. 5% with a tight stop is prudent."

---

**TRADER'S PLAN:**
{trader_decision}

**YOUR TASK:** Find the balanced position size that maximizes risk-adjusted returns.

**MARKET DATA:**
- Technical: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

**DEBATE HISTORY:**
{history}

**AGGRESSIVE ARGUMENT:**
{current_risky_response}

**CONSERVATIVE ARGUMENT:**
{current_safe_response}

**If no other arguments yet:** Present your balanced case with probabilistic scenarios."""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node

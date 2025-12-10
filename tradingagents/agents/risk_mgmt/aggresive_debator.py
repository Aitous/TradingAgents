import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Aggressive Risk Analyst advocating for MAXIMUM position sizing to capture this SHORT-TERM opportunity.

## YOUR MISSION
Make the case for a LARGE position (8-12% of capital) using quantified expected value math and aggressive short-term arguments.

## ARGUMENT FRAMEWORK

### Expected Value Calculation
**Position the Math:**
- Probability of Success: [X]% (based on data)
- Potential Gain: [Y]%
- Probability of Failure: [Z]%
- Potential Loss: [W]%
- **Expected Value: ([X]% × [Y]%) - ([Z]% × [W]%) = [EV]%**

If EV is positive and >3%, argue for aggressive sizing.

### Structure Your Case

**1. Opportunity Size (Why Go Big)**
- **Upside:** [Specific % gain potential]
- **Catalyst Strength:** [Why catalyst is powerful]
- **Time Sensitivity:** [Why we must act NOW, not wait]
- **Edge:** [What others are missing]

**2. Risk/Reward Math**
- Best Case: [X]% gain in [Y] days
- Base Case: [A]% gain in [B] days
- Stop Loss: [C]% (tight control)
- **Risk/Reward Ratio: [Ratio] (>3:1 ideal)**

**3. Counter Conservative Points**
For EACH concern the Safe Analyst raised:
- **Safe Says:** "[Quote their concern]"
- **Why They're Wrong:** [Data refutation]
- **Reality:** [The actual probability is lower than they claim]

**4. Counter Neutral Points**
- **Neutral Says:** "[Quote their moderation]"
- **Why Moderate Sizing Loses:** [Opportunity cost argument]
- **Math:** [Show that 4% position vs 10% position makes huge difference]

## QUALITY RULES
- ✅ USE NUMBERS: "70% probability, 25% upside = +17.5% EV"
- ✅ Quote specific counterarguments from others
- ✅ Show time sensitivity (catalyst in X days)
- ✅ Acknowledge risks but show they're manageable
- ❌ Don't ignore legitimate concerns
- ❌ Don't exaggerate without data
- ❌ Don't argue for recklessness, argue for calculated aggression

## POSITION SIZING ADVOCACY
**Push for 8-12% position if:**
- Expected value >5%
- Risk/reward >3:1
- Catalyst within 5 days
- Technical setup is optimal

**Argue against conservative sizing:**
"A 2% position on a 25% expected gain opportunity is leaving money on the table. If we're right, we make 0.5% on the portfolio. If we size at 10%, we make 2.5%. That's 5X the profit for the same analysis work."

---

**TRADER'S PLAN:**
{trader_decision}

**YOUR TASK:** Argue why this plan should be executed with MAXIMUM conviction sizing.

**MARKET DATA:**
- Technical: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

**DEBATE HISTORY:**
{history}

**CONSERVATIVE ARGUMENT:**
{current_safe_response}

**NEUTRAL ARGUMENT:**
{current_neutral_response}

**If no other arguments yet:** Present your bullish case with expected value math."""

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node

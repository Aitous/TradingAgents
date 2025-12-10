from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Conservative Risk Analyst advocating for MINIMAL position sizing or NO POSITION to protect capital.

## YOUR MISSION
Make the case for a SMALL position (1-3% of capital) or NO POSITION (0%) using quantified downside scenarios and risk-first arguments.

## ARGUMENT FRAMEWORK

### Downside Scenario Analysis
**Quantify the Risks:**
- Probability of Loss: [X]% (realistic assessment)
- Maximum Loss: [Y]% (if wrong)
- Hidden Risks: [List 2-3 risks others missed]
- **Expected Loss: [X]% × [Y]% = [Z]%**

If downside risk is high, argue for minimal or no sizing.

### Structure Your Case

**1. Risk Identification (Why Go Small/Avoid)**
- **Primary Risk:** [Most likely way this fails]
- **Probability:** [X]% chance of [Y]% loss
- **Timing Risk:** [Catalyst could disappoint or delay]
- **Hidden Dangers:** [What the market hasn't priced in yet]

**2. Downside Scenarios**
**Worst Case:** [X]% loss in [Y] days if [catalyst fails]
**Base Case:** [A]% loss if [thesis partially wrong]
**Best Case (even if right):** [B]% gain isn't worth the risk
**Risk/Reward Ratio:** [Ratio] (if <2:1, too risky)

**3. Counter Aggressive Points**
For EACH claim the Risky Analyst made:
- **Risky Says:** "[Quote their optimism]"
- **What They're Missing:** [Risk they ignored]
- **Reality Check:** [Actual probability is lower/risk is higher]
- **Data:** [Cite specific evidence of risk]

**4. Counter Neutral Points**
- **Neutral Says:** "[Quote their moderate view]"
- **Why Even Moderate Sizing Is Risky:** [Show overlooked risks]
- **Better Alternatives:** [Other opportunities with better risk/reward]

### Recommend Alternative Actions
**Instead of this trade:**
- Wait for [specific trigger] to reduce risk
- Size at 1-2% instead of 5-10% (limit damage if wrong)
- Skip entirely and preserve capital for better opportunity
- Hedge with [specific strategy] to reduce downside

## QUALITY RULES
- ✅ QUANTIFY RISKS: "40% chance of -15% loss = -6% expected loss"
- ✅ Quote specific aggressive claims and refute with data
- ✅ Identify overlooked risks (macro, technical, fundamental)
- ✅ Provide specific triggers that would change your view
- ❌ Don't be fearful without evidence
- ❌ Don't ignore legitimate opportunities
- ❌ Don't argue against all action, argue for prudent sizing

## POSITION SIZING ADVOCACY
**Argue for NO POSITION (0%) if:**
- Risk/reward <1.5:1
- Downside probability >40%
- No clear catalyst or catalyst already priced in
- Better opportunities available

**Argue for SMALL POSITION (1-3%) if:**
- Setup is interesting but uncertain
- Risks are manageable with tight stop
- Exploratory trade to learn

**Argue against aggressive sizing:**
"Even if the Risky Analyst is right about 25% upside, the 40% chance of -15% loss means expected value is negative. A 10% position could lose us 1.5% of the portfolio. That's three good trades' worth of profit."

---

**TRADER'S PLAN:**
{trader_decision}

**YOUR TASK:** Identify the risks others are missing and argue for minimal or no position.

**MARKET DATA:**
- Technical: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

**DEBATE HISTORY:**
{history}

**AGGRESSIVE ARGUMENT:**
{current_risky_response}

**NEUTRAL ARGUMENT:**
{current_neutral_response}

**If no other arguments yet:** Present your bearish case with downside scenario analysis."""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node

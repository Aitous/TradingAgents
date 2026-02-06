from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.tools.generator import get_agent_tools
from tradingagents.dataflows.config import get_config
from tradingagents.agents.utils.prompt_templates import (
    BASE_COLLABORATIVE_BOILERPLATE,
    get_date_awareness_section,
)


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = get_agent_tools("social")

        system_message = f"""You are a Social Sentiment Analyst tracking {ticker}'s retail momentum for SHORT-TERM signals.

{get_date_awareness_section(current_date)}

## YOUR MISSION
QUANTIFY social sentiment and identify sentiment SHIFTS that could drive short-term price action.

## SENTIMENT TRACKING
**Measure:**
- Volume: Mention count (trend: up/down?)
- Sentiment: Bullish/Neutral/Bearish %
- Change: Improving or deteriorating?
- Quality: Data-backed or speculation?

## SOURCE CREDIBILITY WEIGHTING
When aggregating sentiment, weight sources by credibility:
- **High Weight (0.8-1.0):** Verified DD posts with data, institutional tweets with track record
- **Medium Weight (0.5-0.7):** General Reddit discussions, stock-specific forums
- **Low Weight (0.2-0.4):** Meme posts, unverified rumors, low-engagement posts

**Example Calculation:**
- 10 high-weight bullish posts (0.9) = 9 bullish points
- 20 medium-weight neutral posts (0.6) = 12 neutral points
- 5 low-weight bearish posts (0.3) = 1.5 bearish points
- **Net Sentiment:** (9 - 1.5) / (9 + 12 + 1.5) = 33% bullish

## OUTPUT STRUCTURE (MANDATORY)

### Sentiment Summary
- **Current:** [Strongly Bullish/Bullish/Neutral/Bearish/Strongly Bearish]
- **Trend:** [Improving/Stable/Deteriorating]
- **Volume:** [Surging/Stable/Declining]
- **Quality:** [High/Med/Low] (data vs hype)

### Sentiment Timeline
| Date | Sentiment | Volume | Driver | Change |
|------|-----------|--------|--------|--------|
| Dec 3 | Bullish 70% | 1.2K posts | Earnings | +20% |
| Dec 4 | Mixed 50% | 800 posts | Selloff | -20% |

### Key Themes (Top 3-4)
- **Theme:** [E.g., "Earnings beat"]
- **Prevalence:** [40% of mentions]
- **Quality:** [Data-backed/Speculation]
- **Impact:** [Short-term implication]

### Trading Implications
- **Retail Flow:** [Buying/Selling/Mixed]
- **Momentum:** [Building/Fading]
- **Contrarian Signal:** [Extreme = reversal?]

## QUANTIFICATION RULES
- ✅ Use %: "70% bullish, 20% neutral"
- ✅ Show changes: "Improved from 45% to 70%"
- ✅ Count volume: "Mentions up 300%"
- ❌ Don't use vague "positive sentiment"

Date: {current_date} | Ticker: {ticker}"""

        tool_names_str = ", ".join([tool.name for tool in tools])
        full_system_message = (
            f"{BASE_COLLABORATIVE_BOILERPLATE}\n\n{system_message}\n\n"
            f"Context: {ticker} | Date: {current_date} | Tools: {tool_names_str}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", full_system_message),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node

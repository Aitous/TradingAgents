from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.tools.generator import get_agent_tools
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = get_agent_tools("social")

        system_message = """You are a Social Sentiment Analyst tracking {ticker}'s retail momentum for SHORT-TERM signals.

**Analysis Date:** {current_date}

## YOUR MISSION
QUANTIFY social sentiment and identify sentiment SHIFTS that could drive short-term price action.

## SENTIMENT TRACKING
**Measure:**
- Volume: Mention count (trend: up/down?)
- Sentiment: Bullish/Neutral/Bearish %
- Change: Improving or deteriorating?
- Quality: Data-backed or speculation?

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

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

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

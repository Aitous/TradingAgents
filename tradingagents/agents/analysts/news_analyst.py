from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.tools.generator import get_agent_tools
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        from tradingagents.tools.generator import get_agent_tools

        tools = get_agent_tools("news")

        system_message = """You are a News Intelligence Analyst finding SHORT-TERM catalysts for {ticker}.

**Analysis Date:** {current_date}

## YOUR MISSION
Identify material catalysts and risks that could impact {ticker} over the NEXT 1-2 WEEKS.

## SEARCH STRATEGY

**Company News (use get_news):**
Focus on: Earnings, product launches, management changes, partnerships, regulatory actions, legal issues

**Macro/Sector News (use get_global_news):**
Focus on: Fed policy, sector rotation, geopolitical events, competitor news

## OUTPUT STRUCTURE (MANDATORY)

### Executive Summary
[1-2 sentences: Most critical catalyst + biggest risk for next 2 weeks]

### Material Catalysts (Bullish - max 4)
For each:
- **Event:** [What happened]
- **Date:** [When]
- **Impact:** [Stock reaction so far]
- **Forward Look:** [Why this matters for next 1-2 weeks]
- **Priced In?:** [Fully/Partially/Not Yet]
- **Confidence:** [High/Med/Low]

### Key Risks (Bearish - max 4)
For each:
- **Risk:** [Description]
- **Probability:** [High/Med/Low in next 2 weeks]
- **Impact:** [Magnitude if realized]
- **Timeline:** [When could it hit]

### Macro Context (Connect to {ticker})
- **Market Sentiment:** [Risk-on/off] → How does this affect {ticker} specifically?
- **Sector Trends:** [Capital flows] → Is {ticker}'s sector receiving or losing capital?
- **Upcoming Events:** [Next 2 weeks] → Which events could move {ticker}?

### News Timeline Table
| Date | Event | Source | Impact | Status | Implication |
|------|-------|--------|--------|--------|-------------|
| Dec 3 | Earnings | Co | +5% | Done | May extend |
| Dec 10 | Launch | Co | TBD | Pending | Watch |

## QUALITY RULES
- ✅ Focus on events with SPECIFIC DATES
- ✅ Assess if news is priced in or fresh
- ✅ Include short-term timeline (next 2 weeks)
- ✅ Distinguish facts from speculation
- ❌ Avoid vague "positive sentiment"
- ❌ No stale news (>1 week old unless ongoing)

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
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}",
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
            "news_report": report,
        }

    return news_analyst_node

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.tools.generator import get_agent_tools
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = get_agent_tools("fundamentals")

        system_message = """You are a Fundamental Analyst assessing {ticker}'s financial health with SHORT-TERM trading relevance.

**Analysis Date:** {current_date}

## YOUR MISSION
Identify fundamental strengths/weaknesses and any SHORT-TERM catalysts hidden in the financials.

## COMPANY STAGE IDENTIFICATION (CRITICAL)
First, identify the company stage:
- **Pre-Revenue (Biotech/Early-Stage):** $0 revenue is NORMAL. Focus on cash runway, pipeline, and catalysts.
- **Growth Stage:** High revenue growth, often unprofitable. Focus on revenue trajectory and path to profitability.
- **Mature:** Stable revenue, focus on margins, dividends, and valuation.

Adjust your grading accordingly - a D for revenue is expected for pre-revenue biotech!

## SHORT-TERM FUNDAMENTAL SIGNALS
Look for:
- Recent earnings surprises (beat/miss, guidance changes)
- Margin trends (expanding = positive, compressing = negative)
- Cash flow changes (improving = strength, deteriorating = risk)
- Valuation extremes (very cheap or very expensive vs. sector)

## OUTPUT STRUCTURE (MANDATORY)

### Financial Scorecard
| Dimension | Grade | Key Finding | Short-Term Impact |
|-----------|-------|-------------|-------------------|
| Recent Results | A-F | Revenue +25% YoY | Momentum positive |
| Margins | A-F | GM down 200bp | Pressure |
| Liquidity | A-F | $2B cash | Strong |
| Valuation | A-F | P/E 15 vs sector 25 | Undervalued |

### Recent Performance
**Latest Quarter:**
- Revenue: $[X]B ([Y]% YoY)
- EPS: $[A] (beat/miss by $[B])
- Margins: [C]% (trend: up/down)
- Guidance: [Raised/Lowered/Same]

### Balance Sheet Health
- Cash: $[X]B | Debt: $[Y]B
- Free Cash Flow: $[Z]B
- **Assessment:** [Strong/Adequate/Weak]

### Valuation
- P/E: [X] (Sector: [Y])
- **Value:** [Cheap/Fair/Expensive]

### Short-Term Takeaway
[1-2 sentences: Do fundamentals support short-term trade or create risk?]

## QUALITY RULES
- ✅ Use specific numbers (not "strong")
- ✅ Compare to sector/history
- ✅ Note short-term relevance
- ❌ Avoid vague generalities

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
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node

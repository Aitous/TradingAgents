import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from tradingagents.dataflows.discovery.discovery_config import DiscoveryConfig
from tradingagents.dataflows.discovery.utils import append_llm_log, resolve_llm_name
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    Extract JSON from markdown code blocks.

    Handles cases where LLMs return JSON wrapped in ```json...``` or just ```...```
    """
    if not text:
        return None

    # Try to find JSON in markdown code blocks
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If no code blocks, check if the text itself is valid JSON
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text

    return None


class StockRanking(BaseModel):
    """Single stock ranking."""

    rank: int = Field(description="Rank 1-N")
    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Company name")
    current_price: float = Field(description="Current stock price")
    strategy_match: str = Field(description="Strategy that matched")
    final_score: int = Field(description="Score 0-100")
    confidence: int = Field(description="Confidence 1-10")
    reason: str = Field(
        description="Detailed investment thesis (4-6 sentences) defending the trade with specific catalysts, risk/reward, and timing"
    )
    description: str = Field(description="Company description")


class RankingResponse(BaseModel):
    """LLM ranking response."""

    rankings: List[StockRanking] = Field(description="List of ranked stocks")


class CandidateRanker:
    """
    Handles ranking of filtered candidates using Deep Thinking LLM.
    """

    def __init__(self, config: Dict[str, Any], llm: BaseChatModel, analytics: Any):
        self.config = config
        self.llm = llm
        self.analytics = analytics

        dc = DiscoveryConfig.from_config(config)
        self.max_candidates_to_analyze = dc.ranker.max_candidates_to_analyze
        self.final_recommendations = dc.ranker.final_recommendations

        # Truncation settings
        self.truncate_context = dc.ranker.truncate_ranking_context
        self.max_news_chars = dc.ranker.max_news_chars
        self.max_insider_chars = dc.ranker.max_insider_chars
        self.max_recommendations_chars = dc.ranker.max_recommendations_chars

        # Prompt logging
        self.log_prompts_console = dc.logging.log_prompts_console

    def rank(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Rank all filtered candidates and select the top opportunities."""
        candidates = state.get("candidate_metadata", [])
        trade_date = state.get("trade_date", datetime.now().strftime("%Y-%m-%d"))

        if len(candidates) == 0:
            logger.warning("⚠️ No candidates to rank.")
            return {
                "opportunities": [],
                "final_ranking": "[]",
                "status": "complete",
                "tool_logs": state.get("tool_logs", []),
            }

        # Limit candidates to prevent token overflow
        max_candidates = min(self.max_candidates_to_analyze, 200)
        if len(candidates) > max_candidates:
            logger.warning(
                f"⚠️ Too many candidates ({len(candidates)}), limiting to top {max_candidates} by priority"
            )
            candidates = candidates[:max_candidates]

        logger.info(
            f"🏆 Ranking {len(candidates)} candidates to select top {self.final_recommendations}..."
        )

        # Load historical performance statistics
        historical_stats = self.analytics.load_historical_stats()
        if historical_stats.get("available"):
            logger.info(
                f"📊 Loaded historical stats: {historical_stats.get('total_tracked', 0)} tracked recommendations"
            )

        # Build RICH context for each candidate
        candidate_summaries = []
        for cand in candidates:
            ticker = cand.get("ticker", "UNKNOWN")
            strategy = cand.get("strategy", "unknown")
            priority = cand.get("priority", "unknown")
            context = cand.get("context", "No context available")
            all_sources = cand.get("all_sources", [cand.get("source", "unknown")])
            technical_indicators = cand.get("technical_indicators", "")
            avg_volume = cand.get("average_volume", "N/A")
            intraday_change = cand.get("intraday_change_pct", "N/A")
            current_price = cand.get("current_price")

            # Formatting helpers
            volume_str = (
                f"{avg_volume:,.0f}" if isinstance(avg_volume, (int, float)) else str(avg_volume)
            )
            intraday_str = (
                f"{intraday_change:+.1f}%"
                if isinstance(intraday_change, (int, float))
                else str(intraday_change)
            )
            price_str = f"${current_price:.2f}" if current_price else "N/A"

            # Use fundamentals already fetched - pass more complete data
            fund = cand.get("fundamentals", {})
            fundamentals_summary = self._format_fundamentals_expanded(fund)

            # Use full technical indicators instead of extracting only RSI
            tech_summary = (
                technical_indicators if technical_indicators else "No technical data available."
            )

            # Get options activity
            options_activity = cand.get("options_activity", "")

            # Get business description for context
            business_description = cand.get("business_description", "")

            # News summary - handle both batch news (string) and discovery news (list of dicts)
            news_items = cand.get("news", [])
            news_summary = ""
            if isinstance(news_items, list) and news_items:
                # List format from discovery scanner
                headlines = []
                for item in news_items[:3]:
                    if isinstance(item, dict):
                        # Discovery news format: {'news_title': '...', 'news_summary': '...', 'sentiment': '...', 'published_at': '...'}
                        title = item.get("news_title", item.get("title", ""))
                        summary = item.get("news_summary", "")
                        # Get timestamp from various possible fields
                        timestamp = item.get("published_at") or item.get("timestamp") or ""
                        # Format timestamp for display (extract date/time portion)
                        time_str = self._format_news_timestamp(timestamp)
                        if title:
                            if time_str:
                                headlines.append(
                                    f"[{time_str}] {title}: {summary}"
                                    if summary
                                    else f"[{time_str}] {title}"
                                )
                            else:
                                headlines.append(f"{title}: {summary}" if summary else title)
                    elif isinstance(item, str):
                        headlines.append(item)
                news_summary = "; ".join(headlines) if headlines else ""
            elif isinstance(news_items, str):
                news_summary = news_items

            # Apply truncation if configured
            if self.truncate_context and self.max_news_chars > 0:
                if len(news_summary) > self.max_news_chars:
                    news_summary = news_summary[: self.max_news_chars] + "..."

            source_str = (
                ", ".join(all_sources) if isinstance(all_sources, list) else str(all_sources)
            )

            # Format insider/analyst data
            insider_text = cand.get("insider_transactions", "N/A")
            recommendations_text = cand.get("recommendations", "N/A")

            # Apply truncation if configured
            if self.truncate_context:
                if (
                    self.max_insider_chars > 0
                    and isinstance(insider_text, str)
                    and len(insider_text) > self.max_insider_chars
                ):
                    insider_text = insider_text[: self.max_insider_chars] + "..."
                if (
                    self.max_recommendations_chars > 0
                    and isinstance(recommendations_text, str)
                    and len(recommendations_text) > self.max_recommendations_chars
                ):
                    recommendations_text = (
                        recommendations_text[: self.max_recommendations_chars] + "..."
                    )

            # New enrichment fields
            confluence_score = cand.get("confluence_score", 1)
            quant_score = cand.get("quant_score", "N/A")

            # ML prediction
            ml_win_prob = cand.get("ml_win_probability")
            ml_prediction = cand.get("ml_prediction")
            if ml_win_prob is not None:
                ml_str = f"{ml_win_prob:.1%} (Predicted: {ml_prediction})"
            else:
                ml_str = "N/A"
            short_interest_pct = cand.get("short_interest_pct")
            high_short = cand.get("high_short_interest", False)
            short_str = f"{short_interest_pct:.1f}%" if short_interest_pct else "N/A"
            if high_short:
                short_str += " (HIGH)"

            # Earnings estimate
            if cand.get("has_upcoming_earnings"):
                days = cand.get("days_to_earnings", "?")
                eps_est = cand.get("eps_estimate")
                rev_est = cand.get("revenue_estimate")
                earnings_date = cand.get("earnings_date", "N/A")
                eps_str = f"${eps_est:.2f}" if isinstance(eps_est, (int, float)) else "N/A"
                rev_str = f"${rev_est:,.0f}" if isinstance(rev_est, (int, float)) else "N/A"
                earnings_section = f"Earnings in {days} days ({earnings_date}): EPS Est {eps_str}, Rev Est {rev_str}"
            else:
                earnings_section = "No upcoming earnings within 30 days"

            summary = f"""### {ticker} (Priority: {priority.upper()})
- **Strategy Match**: {strategy}
- **Sources**: {source_str} | **Confluence**: {confluence_score} source(s)
- **Quant Pre-Score**: {quant_score}/100 | **ML Win Probability**: {ml_str}
- **Price**: {price_str} | **Current Price (numeric)**: {current_price if isinstance(current_price, (int, float)) else "N/A"} | **Intraday**: {intraday_str} | **Avg Volume**: {volume_str}
- **Short Interest**: {short_str}
- **Discovery Context**: {context}
- **Business**: {business_description}
- **News**: {news_summary}

**Technical Analysis**:
{tech_summary}

**Fundamentals**: {fundamentals_summary}

**Insider Transactions**:
{insider_text}

**Analyst Recommendations**:
{recommendations_text}

**Options Activity**:
{options_activity if options_activity else "N/A"}

**Upcoming Earnings**: {earnings_section}
"""
            candidate_summaries.append(summary)

        combined_candidates_text = "\n".join(candidate_summaries)

        # Build Prompt
        prompt = f"""You are an analyst tasked with selecting the absolute best {self.final_recommendations} stock opportunities from a pre-filtered list.

CURRENT DATE: {trade_date}

GOAL: Select the top {self.final_recommendations} stocks with the highest probability of generating >5% returns in the next 1-7 days.
Focus on asymmetric risk/reward: massive upside potential with managed risk.

HISTORICAL INSIGHTS:
{json.dumps(historical_stats.get('summary', 'N/A'), indent=2)}

CANDIDATES FOR REVIEW:
{combined_candidates_text}

INSTRUCTIONS:
1. Analyze each candidate's "Discovery Context" (why it was found) and "Strategy Match".
2. Cross-reference with Technicals (RSI, etc.) and Fundamentals.
3. Use the Quantitative Pre-Score as an objective baseline. Scores above 50 indicate strong multi-factor alignment.
4. The ML Win Probability is a trained model's estimate that this stock hits +5% within 7 days. Treat scores above 60% as strong ML confirmation.
5. Prioritize "LEADING" indicators (Undiscovered DD, Earnings Accumulation, Insider Buying) over lagging ones.
6. Select exactly {self.final_recommendations} winners.
7. Use ONLY the information provided in the candidates section; do NOT invent catalysts, prices, or metrics.
8. If a required field is missing, set it to null (do not guess).
9. Rank only tickers from the candidates list.
10. Reasons must reference at least two concrete facts from the candidate context.

Output a JSON object with a 'rankings' list. Each item should have:
- rank: 1 to {self.final_recommendations}
- ticker: stock symbol
- company_name: name
- current_price: price
- strategy_match: main strategy
- final_score: 0-100 score
- confidence: 1-10 confidence level
- reason: Detailed investment thesis (4-6 sentences). Defend the trade: (1) what is the catalyst/edge, (2) why NOW and not later, (3) what does the risk/reward look like, (4) what could go wrong. Reference specific data points from the candidate context.
- description: Brief company description.

JSON FORMAT ONLY. No markdown, no extra text. All numeric fields must be numbers (not strings)."""

        # Invoke LLM with structured output
        logger.info("🧠 Deep Thinking Ranker analyzing opportunities...")
        logger.info(
            f"Invoking ranking LLM with {len(candidates)} candidates, prompt length: {len(prompt)} chars"
        )
        if self.log_prompts_console:
            logger.info(f"Full ranking prompt:\n{prompt}")
        else:
            logger.debug(f"Full ranking prompt:\n{prompt}")

        try:
            # Use structured output with include_raw for debugging
            structured_llm = self.llm.with_structured_output(RankingResponse, include_raw=True)
            response = structured_llm.invoke([HumanMessage(content=prompt)])

            tool_logs = state.get("tool_logs", [])
            append_llm_log(
                tool_logs,
                node="ranker",
                step="Rank candidates",
                model=resolve_llm_name(self.llm),
                prompt=prompt,
                output=response,
            )
            state["tool_logs"] = tool_logs

            # Handle the response (dict with raw, parsed, parsing_error)
            if isinstance(response, dict):
                result = response.get("parsed")
                raw = response.get("raw")
                parsing_error = response.get("parsing_error")

                # Log debug info
                logger.info(f"Structured output - parsed type: {type(result)}")
                if parsing_error:
                    logger.error(f"Parsing error: {parsing_error}")
                if raw and hasattr(raw, "content"):
                    logger.debug(f"Raw content preview: {str(raw.content)[:500]}...")
            else:
                # Direct RankingResponse (shouldn't happen with include_raw=True)
                result = response

            # Extract rankings - with fallback for markdown-wrapped JSON
            if result is None:
                logger.warning(
                    "Structured output parsing returned None - attempting fallback extraction"
                )

                # Try to extract JSON from raw response (handles ```json...``` wrapping)
                raw_text = None
                if raw and hasattr(raw, "content"):
                    content = raw.content
                    if isinstance(content, str):
                        raw_text = content
                    elif isinstance(content, list):
                        # Handle list of content blocks (e.g., [{'type': 'text', 'text': '...'}])
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                raw_text = block.get("text", "")
                                break
                            elif isinstance(block, str):
                                raw_text = block
                                break

                if raw_text:
                    json_str = extract_json_from_markdown(raw_text)
                    if json_str:
                        try:
                            parsed_data = json.loads(json_str)
                            result = RankingResponse.model_validate(parsed_data)
                            logger.info(
                                "Successfully extracted JSON from markdown-wrapped response"
                            )
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse extracted JSON: {e}")
                        except Exception as e:
                            logger.error(f"Failed to validate extracted JSON: {e}")

                if result is None:
                    logger.error("Parsed result is None - check raw response for clues")
                    raise ValueError(
                        "LLM returned None. This may be due to content filtering or prompt length. "
                        "Check LOG_LEVEL=DEBUG for details."
                    )

            if not hasattr(result, "rankings"):
                logger.error(f"Result missing 'rankings'. Type: {type(result)}, Value: {result}")
                raise ValueError(f"Unexpected result format: {type(result)}")

            final_ranking_list = [ranking.model_dump() for ranking in result.rankings]

            logger.info(f"✅ Selected {len(final_ranking_list)} top recommendations")
            logger.info(
                f"Successfully ranked {len(final_ranking_list)} opportunities: "
                f"{[r['ticker'] for r in final_ranking_list]}"
            )

            # Update state with opportunities for downstream use (deep dive)
            state_opportunities = []
            for rank_dict in final_ranking_list:
                ticker = rank_dict["ticker"].upper()
                # Find original candidate metadata
                meta = next((c for c in candidates if c.get("ticker") == ticker), {})

                state_opportunities.append(
                    {
                        "ticker": ticker,
                        "strategy": rank_dict["strategy_match"],
                        "reason": rank_dict["reason"],
                        "score": rank_dict["final_score"],
                        "rank": rank_dict["rank"],
                        "metadata": meta,
                    }
                )

            return {
                "final_ranking": final_ranking_list,  # List of dicts
                "opportunities": state_opportunities,
                "status": "ranked",
            }

        except ValueError as e:
            tool_logs = state.get("tool_logs", [])
            append_llm_log(
                tool_logs,
                node="ranker",
                step="Rank candidates",
                model=resolve_llm_name(self.llm),
                prompt=prompt,
                output="",
                error=str(e),
            )
            state["tool_logs"] = tool_logs
            # Structured output validation failed
            logger.error(f"❌ Error: {e}")
            logger.error(f"Structured output validation error: {e}")
            return {"final_ranking": [], "opportunities": [], "status": "ranking_failed"}

        except Exception as e:
            tool_logs = state.get("tool_logs", [])
            append_llm_log(
                tool_logs,
                node="ranker",
                step="Rank candidates",
                model=resolve_llm_name(self.llm),
                prompt=prompt,
                output="",
                error=str(e),
            )
            state["tool_logs"] = tool_logs
            logger.error(f"❌ Error during ranking: {e}")
            logger.exception(f"Unexpected error during ranking: {e}")
            return {"final_ranking": [], "opportunities": [], "status": "error"}

    def _format_news_timestamp(self, timestamp: str) -> str:
        """
        Format news timestamp for display in ranking prompt.

        Handles various timestamp formats:
        - ISO-8601: 2026-01-31T14:30:00Z -> Jan 31 14:30
        - Date only: 2026-01-31 -> Jan 31
        - Already formatted strings pass through
        """
        if not timestamp:
            return ""

        try:
            # Try ISO-8601 format first
            if "T" in timestamp:
                # Parse ISO format: 2026-01-31T14:30:00Z or 2026-01-31T14:30:00+00:00
                dt_str = timestamp.replace("Z", "+00:00")
                # Handle timezone suffix
                if "+" in dt_str:
                    dt_str = dt_str.split("+")[0]
                elif dt_str.count("-") > 2:
                    # Handle negative timezone offset like -05:00
                    parts = dt_str.rsplit("-", 1)
                    if ":" in parts[-1]:
                        dt_str = parts[0]

                dt = datetime.fromisoformat(dt_str)
                return dt.strftime("%b %d %H:%M")

            # Try date-only format
            if len(timestamp) == 10 and timestamp.count("-") == 2:
                dt = datetime.strptime(timestamp, "%Y-%m-%d")
                return dt.strftime("%b %d")

            # Try compact format from Alpha Vantage: 20260131T143000
            if len(timestamp) >= 8 and timestamp[:8].isdigit():
                dt = datetime.strptime(timestamp[:8], "%Y%m%d")
                if len(timestamp) >= 15 and timestamp[8] == "T":
                    dt = datetime.strptime(timestamp[:15], "%Y%m%dT%H%M%S")
                    return dt.strftime("%b %d %H:%M")
                return dt.strftime("%b %d")

            # If it's already a short readable format, return as-is
            if len(timestamp) <= 20:
                return timestamp

        except (ValueError, AttributeError):
            # If parsing fails, return empty to avoid cluttering output
            pass

        return ""

    def _format_fundamentals_expanded(self, fund: Dict[str, Any]) -> str:
        """Format fundamentals dictionary with comprehensive data for ranking LLM."""
        if not fund:
            return "N/A"

        def fmt_pct(val):
            if val == "N/A" or val is None:
                return "N/A"
            try:
                return f"{float(val)*100:.1f}%"
            except Exception:
                return str(val)

        def fmt_large(val, prefix="$"):
            if val == "N/A" or val is None:
                return "N/A"
            try:
                n = float(val)
                if n >= 1e12:
                    return f"{prefix}{n/1e12:.2f}T"
                if n >= 1e9:
                    return f"{prefix}{n/1e9:.2f}B"
                if n >= 1e6:
                    return f"{prefix}{n/1e6:.1f}M"
                return f"{prefix}{n:,.0f}"
            except Exception:
                return str(val)

        def fmt_ratio(val):
            if val == "N/A" or val is None:
                return "N/A"
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val)

        parts = []

        # Basic info
        sector = fund.get("Sector", "N/A")
        industry = fund.get("Industry", "N/A")
        if sector != "N/A":
            parts.append(f"Sector: {sector}")
        if industry != "N/A":
            parts.append(f"Industry: {industry}")

        # Valuation
        mc = fmt_large(fund.get("MarketCapitalization"))
        pe = fmt_ratio(fund.get("PERatio"))
        fwd_pe = fmt_ratio(fund.get("ForwardPE"))
        peg = fmt_ratio(fund.get("PEGRatio"))
        pb = fmt_ratio(fund.get("PriceToBookRatio"))
        ps = fmt_ratio(fund.get("PriceToSalesRatioTTM"))

        valuation_parts = []
        if mc != "N/A":
            valuation_parts.append(f"Cap: {mc}")
        if pe != "N/A":
            valuation_parts.append(f"P/E: {pe}")
        if fwd_pe != "N/A":
            valuation_parts.append(f"Fwd P/E: {fwd_pe}")
        if peg != "N/A":
            valuation_parts.append(f"PEG: {peg}")
        if pb != "N/A":
            valuation_parts.append(f"P/B: {pb}")
        if ps != "N/A":
            valuation_parts.append(f"P/S: {ps}")
        if valuation_parts:
            parts.append("Valuation: " + ", ".join(valuation_parts))

        # Growth metrics
        rev_growth = fmt_pct(fund.get("QuarterlyRevenueGrowthYOY"))
        earnings_growth = fmt_pct(fund.get("QuarterlyEarningsGrowthYOY"))

        growth_parts = []
        if rev_growth != "N/A":
            growth_parts.append(f"Rev Growth: {rev_growth}")
        if earnings_growth != "N/A":
            growth_parts.append(f"Earnings Growth: {earnings_growth}")
        if growth_parts:
            parts.append("Growth: " + ", ".join(growth_parts))

        # Profitability
        profit_margin = fmt_pct(fund.get("ProfitMargin"))
        oper_margin = fmt_pct(fund.get("OperatingMarginTTM"))
        roe = fmt_pct(fund.get("ReturnOnEquityTTM"))
        roa = fmt_pct(fund.get("ReturnOnAssetsTTM"))

        profit_parts = []
        if profit_margin != "N/A":
            profit_parts.append(f"Profit Margin: {profit_margin}")
        if oper_margin != "N/A":
            profit_parts.append(f"Oper Margin: {oper_margin}")
        if roe != "N/A":
            profit_parts.append(f"ROE: {roe}")
        if roa != "N/A":
            profit_parts.append(f"ROA: {roa}")
        if profit_parts:
            parts.append("Profitability: " + ", ".join(profit_parts))

        # Dividend info
        div_yield = fmt_pct(fund.get("DividendYield"))
        if div_yield != "N/A" and div_yield != "0.0%":
            parts.append(f"Dividend: {div_yield} yield")

        # Financial health
        current_ratio = fmt_ratio(fund.get("CurrentRatio"))
        debt_to_equity = fmt_ratio(fund.get("DebtToEquity"))
        if current_ratio != "N/A" or debt_to_equity != "N/A":
            health_parts = []
            if current_ratio != "N/A":
                health_parts.append(f"Current Ratio: {current_ratio}")
            if debt_to_equity != "N/A":
                health_parts.append(f"D/E: {debt_to_equity}")
            parts.append("Financial Health: " + ", ".join(health_parts))

        # Analyst targets
        target_high = fmt_large(fund.get("AnalystTargetPrice"))
        if target_high != "N/A":
            parts.append(f"Analyst Target: {target_high}")

        # Earnings info
        eps = fund.get("EPS", "N/A")
        if eps != "N/A":
            try:
                eps = f"${float(eps):.2f}"
                parts.append(f"EPS: {eps}")
            except Exception:
                pass

        # Beta (volatility)
        beta = fund.get("Beta", "N/A")
        if beta != "N/A":
            try:
                beta = f"{float(beta):.2f}"
                parts.append(f"Beta: {beta}")
            except Exception:
                pass

        # 52-week range
        week52_high = fund.get("52WeekHigh", "N/A")
        week52_low = fund.get("52WeekLow", "N/A")
        if week52_high != "N/A" and week52_low != "N/A":
            try:
                parts.append(f"52W Range: ${float(week52_low):.2f} - ${float(week52_high):.2f}")
            except Exception:
                pass

        # Short interest
        short_pct = fund.get("ShortPercentFloat", "N/A")
        if short_pct != "N/A":
            try:
                parts.append(f"Short Interest: {float(short_pct)*100:.1f}%")
            except Exception:
                pass

        return " | ".join(parts) if parts else "N/A"

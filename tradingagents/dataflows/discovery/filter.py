import json
import re
from datetime import timedelta
from typing import Any, Callable, Dict, List

from tradingagents.dataflows.discovery.candidate import Candidate
from tradingagents.dataflows.discovery.utils import (
    PRIORITY_ORDER,
    Strategy,
    is_valid_ticker,
    resolve_trade_date,
)


def _parse_market_cap_to_billions(value: Any) -> Any:
    """Parse market cap into billions of USD when possible."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        # Assume raw dollars if large; otherwise already in billions
        return round(value / 1_000_000_000, 3) if value > 1_000_000 else float(value)

    if isinstance(value, str):
        text = value.strip().upper().replace(",", "").replace("$", "")
        if not text or text in {"N/A", "NA", "NONE"}:
            return None

        multipliers = {"T": 1000.0, "B": 1.0, "M": 0.001, "K": 0.000001}
        suffix = text[-1]
        if suffix in multipliers:
            try:
                return round(float(text[:-1]) * multipliers[suffix], 3)
            except ValueError:
                return None

        # Fallback: treat as raw dollars
        try:
            numeric = float(text)
            return round(numeric / 1_000_000_000, 3) if numeric > 1_000_000 else numeric
        except ValueError:
            return None

    return None


def _extract_atr_pct(technical_report: str) -> Any:
    """Extract ATR % of price from technical report."""
    if not technical_report:
        return None
    match = re.search(r"ATR:\s*\$?[\d\.]+\s*\(([\d\.]+)% of price\)", technical_report)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_bb_width_pct(technical_report: str) -> Any:
    """Extract Bollinger bandwidth % from technical report."""
    if not technical_report:
        return None
    match = re.search(r"Bandwidth:\s*([\d\.]+)%", technical_report)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _build_combined_context(
    primary_context: str,
    context_details: list,
    max_snippets: int,
    snippet_max_chars: int,
) -> str:
    """Combine multiple contexts into a compact summary."""
    if not context_details:
        return primary_context or ""

    primary_context = primary_context or context_details[0]
    others = [c for c in context_details if c and c != primary_context]
    if not others:
        return primary_context

    trimmed = []
    for item in others[:max_snippets]:
        snippet = item.strip()
        if len(snippet) > snippet_max_chars:
            snippet = snippet[:snippet_max_chars].rstrip() + "..."
        trimmed.append(snippet)

    if not trimmed:
        return primary_context

    return f"{primary_context} | Other signals: " + "; ".join(trimmed)


class CandidateFilter:
    """
    Handles filtering and enrichment of discovery candidates.
    """

    def __init__(self, config: Dict[str, Any], tool_executor: Callable):
        self.config = config
        self.execute_tool = tool_executor

        # Discovery Settings
        discovery_config = config.get("discovery", {})

        # Filter settings (nested under "filters" section, with backward compatibility)
        filter_config = discovery_config.get("filters", discovery_config)  # Fallback to root for old configs
        self.filter_same_day_movers = filter_config.get("filter_same_day_movers", True)
        self.intraday_movement_threshold = filter_config.get("intraday_movement_threshold", 10.0)
        self.filter_recent_movers = filter_config.get("filter_recent_movers", True)
        self.recent_movement_lookback_days = filter_config.get("recent_movement_lookback_days", 7)
        self.recent_movement_threshold = filter_config.get("recent_movement_threshold", 10.0)
        self.recent_mover_action = filter_config.get("recent_mover_action", "filter")
        self.min_average_volume = filter_config.get("min_average_volume", 500_000)
        self.volume_lookback_days = filter_config.get("volume_lookback_days", 10)

        # Enrichment settings (nested under "enrichment" section, with backward compatibility)
        enrichment_config = discovery_config.get("enrichment", discovery_config)  # Fallback to root
        self.batch_news_vendor = enrichment_config.get("batch_news_vendor", "openai")
        self.batch_news_batch_size = enrichment_config.get("batch_news_batch_size", 50)

        # Other settings (remain at discovery level)
        self.news_lookback_days = discovery_config.get("news_lookback_days", 3)
        self.volume_cache_key = discovery_config.get("volume_cache_key", "avg_volume_cache")
        self.min_market_cap = discovery_config.get("min_market_cap", 0)
        self.compression_atr_pct_max = discovery_config.get("compression_atr_pct_max", 2.0)
        self.compression_bb_width_max = discovery_config.get("compression_bb_width_max", 6.0)
        self.compression_min_volume_ratio = discovery_config.get("compression_min_volume_ratio", 1.3)
        self.context_max_snippets = discovery_config.get("context_max_snippets", 2)
        self.context_snippet_max_chars = discovery_config.get("context_snippet_max_chars", 140)

    def filter(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Filter candidates based on strategy and enrich with additional data."""
        candidates = state.get("candidate_metadata", [])
        if not candidates:
            # Fallback if metadata missing (backward compatibility)
            candidates = [{"ticker": t, "source": "unknown"} for t in state["tickers"]]

        # Calculate date range for news (configurable days back from trade_date)
        end_date_obj = resolve_trade_date(state)

        start_date_obj = end_date_obj - timedelta(days=self.news_lookback_days)
        start_date = start_date_obj.strftime("%Y-%m-%d")
        end_date = end_date_obj.strftime("%Y-%m-%d")

        print(f"🔍 Filtering and enriching {len(candidates)} candidates...")

        priority_order = self._priority_order()
        candidates = self._dedupe_candidates(candidates, priority_order)
        candidates = self._sort_by_priority(candidates, priority_order)
        self._log_priority_breakdown(candidates)

        volume_by_ticker = self._fetch_batch_volume(state, candidates)
        news_by_ticker = self._fetch_batch_news(start_date, end_date, candidates)

        (
            filtered_candidates,
            filtered_reasons,
            failed_tickers,
            delisted_cache,
        ) = self._filter_and_enrich_candidates(
            state=state,
            candidates=candidates,
            volume_by_ticker=volume_by_ticker,
            news_by_ticker=news_by_ticker,
            end_date=end_date,
        )

        # Print consolidated filtering summary
        self._print_filter_summary(candidates, filtered_candidates, filtered_reasons)

        # Print consolidated list of failed tickers
        if failed_tickers:
            print(f"\n   ⚠️  {len(failed_tickers)} tickers failed data fetch (possibly delisted)")
            if len(failed_tickers) <= 10:
                print(f"      {', '.join(failed_tickers)}")
            else:
                print(
                    f"      {', '.join(failed_tickers[:10])} ... and {len(failed_tickers)-10} more"
                )
            # Export review list
            delisted_cache.export_review_list()

        return {
            "filtered_tickers": [c["ticker"] for c in filtered_candidates],
            "candidate_metadata": filtered_candidates,
            "status": "filtered",
        }

    def _priority_order(self) -> Dict[str, int]:
        return dict(PRIORITY_ORDER)

    def _dedupe_candidates(
        self, candidates: List[Dict[str, Any]], priority_order: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Deduplicate by ticker while preserving multi-source evidence."""
        unique_candidates: Dict[str, Candidate] = {}

        for cand in candidates:
            ticker = cand.get("ticker")
            if not ticker or not is_valid_ticker(ticker):
                continue

            candidate = Candidate.from_dict(cand)
            ticker = candidate.ticker

            if ticker not in unique_candidates:
                unique_candidates[ticker] = candidate
                continue

            existing = unique_candidates[ticker]
            existing_rank = priority_order.get(existing.priority, 4)
            incoming_rank = priority_order.get(candidate.priority, 4)

            if incoming_rank < existing_rank:
                primary = candidate
                secondary = existing
            elif incoming_rank == existing_rank:
                existing_context = existing.context
                incoming_context = candidate.context
                if len(incoming_context) > len(existing_context):
                    primary = candidate
                    secondary = existing
                else:
                    primary = existing
                    secondary = candidate
            else:
                primary = existing
                secondary = candidate

            # Merge sources and contexts
            merged_sources = list(dict.fromkeys(primary.all_sources + secondary.all_sources))
            merged_contexts = list(
                dict.fromkeys(primary.context_details + secondary.context_details)
            )

            primary.all_sources = merged_sources
            primary.context_details = merged_contexts
            primary.context = _build_combined_context(
                primary.context,
                merged_contexts,
                max_snippets=self.context_max_snippets,
                snippet_max_chars=self.context_snippet_max_chars,
            )

            if secondary.allow_invalid:
                primary.allow_invalid = True

            unique_candidates[ticker] = primary

        return [candidate.to_dict() for candidate in unique_candidates.values()]

    def _sort_by_priority(
        self, candidates: List[Dict[str, Any]], priority_order: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        candidates.sort(key=lambda x: priority_order.get(x.get("priority", "unknown"), 4))
        return candidates

    def _log_priority_breakdown(self, candidates: List[Dict[str, Any]]) -> None:
        critical_priority = sum(1 for c in candidates if c.get("priority") == "critical")
        high_priority = sum(1 for c in candidates if c.get("priority") == "high")
        medium_priority = sum(1 for c in candidates if c.get("priority") == "medium")
        low_priority = sum(1 for c in candidates if c.get("priority") == "low")
        print(
            f"   Priority breakdown: {critical_priority} critical, {high_priority} high, {medium_priority} medium, {low_priority} low"
        )

    def _fetch_batch_volume(
        self, state: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not (self.min_average_volume and candidates):
            return {}
        return self._run_tool(
            state=state,
            step="Check average volume (batch)",
            tool_name="get_average_volume_batch",
            default={},
            symbols=[c.get("ticker", "") for c in candidates],
            lookback_days=self.volume_lookback_days,
            curr_date=state.get("trade_date"),
            cache_key=self.volume_cache_key,
        )

    def _fetch_batch_news(
        self, start_date: str, end_date: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        all_tickers = [c.get("ticker", "") for c in candidates if c.get("ticker")]
        if not all_tickers:
            return {}

        try:
            if self.batch_news_vendor == "google":
                from tradingagents.dataflows.openai import get_batch_stock_news_google

                print(f"   📰 Batch fetching news (Google) for {len(all_tickers)} tickers...")
                news_by_ticker = self._run_call(
                    "batch fetching news (Google)",
                    get_batch_stock_news_google,
                    default={},
                    tickers=all_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    batch_size=self.batch_news_batch_size,
                )
            else:  # Default to OpenAI
                from tradingagents.dataflows.openai import get_batch_stock_news_openai

                print(f"   📰 Batch fetching news (OpenAI) for {len(all_tickers)} tickers...")
                news_by_ticker = self._run_call(
                    "batch fetching news (OpenAI)",
                    get_batch_stock_news_openai,
                    default={},
                    tickers=all_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    batch_size=self.batch_news_batch_size,
                )
            print(f"   ✓ Batch news fetched for {len(news_by_ticker)} tickers")
            return news_by_ticker
        except Exception as e:
            print(f"   Warning: Batch news fetch failed, will skip news enrichment: {e}")
            return {}

    def _filter_and_enrich_candidates(
        self,
        state: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        volume_by_ticker: Dict[str, Any],
        news_by_ticker: Dict[str, Any],
        end_date: str,
    ):
        filtered_candidates = []
        filtered_reasons = {
            "volume": 0,
            "intraday_moved": 0,
            "recent_moved": 0,
            "market_cap": 0,
            "no_data": 0,
        }

        # Initialize delisted cache for tracking failed tickers
        from tradingagents.dataflows.delisted_cache import DelistedCache

        delisted_cache = DelistedCache()
        failed_tickers = []

        for cand in candidates:
            ticker = cand["ticker"]

            try:
                # Same-day mover filter (check intraday movement first)
                if self.filter_same_day_movers:
                    from tradingagents.dataflows.y_finance import check_intraday_movement

                    try:
                        intraday_check = check_intraday_movement(
                            ticker=ticker, movement_threshold=self.intraday_movement_threshold
                        )

                        # Skip if already moved significantly today
                        if intraday_check.get("already_moved"):
                            filtered_reasons["intraday_moved"] += 1
                            intraday_pct = intraday_check.get("intraday_change_pct", 0)
                            print(
                                f"   Filtered {ticker}: Already moved {intraday_pct:+.1f}% today (stale)"
                            )
                            continue

                        # Add intraday data to candidate metadata for ranking
                        cand["intraday_change_pct"] = intraday_check.get("intraday_change_pct", 0)

                    except Exception as e:
                        # Don't filter out if check fails, just log
                        print(f"   Warning: Could not check intraday movement for {ticker}: {e}")

                # Recent multi-day mover filter (avoid stocks that already ran)
                if self.filter_recent_movers:
                    from tradingagents.dataflows.y_finance import check_if_price_reacted

                    try:
                        reaction = check_if_price_reacted(
                            ticker=ticker,
                            lookback_days=self.recent_movement_lookback_days,
                            reaction_threshold=self.recent_movement_threshold,
                        )
                        cand["recent_change_pct"] = reaction.get("price_change_pct")
                        cand["recent_move_status"] = reaction.get("status")

                        if reaction.get("status") == "lagging":
                            if self.recent_mover_action == "filter":
                                filtered_reasons["recent_moved"] += 1
                                change_pct = reaction.get("price_change_pct", 0)
                                print(
                                    f"   Filtered {ticker}: Already moved {change_pct:+.1f}% in last "
                                    f"{self.recent_movement_lookback_days} days"
                                )
                                continue
                            if self.recent_mover_action == "deprioritize":
                                cand["priority"] = "low"
                                existing_context = cand.get("context", "")
                                change_pct = reaction.get("price_change_pct", 0)
                                cand["context"] = (
                                    f"{existing_context} | ⚠️ Recent move: {change_pct:+.1f}% "
                                    f"over {self.recent_movement_lookback_days}d"
                                )
                    except Exception as e:
                        print(f"   Warning: Could not check recent movement for {ticker}: {e}")

                # Liquidity filter based on average volume
                if self.min_average_volume:
                    volume_data = {}
                    if isinstance(volume_by_ticker, dict):
                        volume_data = volume_by_ticker.get(ticker.upper(), {})
                    avg_volume = None
                    latest_volume = None
                    if isinstance(volume_data, dict):
                        avg_volume = volume_data.get("average_volume")
                        latest_volume = volume_data.get("latest_volume")
                    elif isinstance(volume_data, (int, float)):
                        avg_volume = float(volume_data)
                    cand["average_volume"] = avg_volume
                    cand["latest_volume"] = latest_volume

                    if avg_volume and latest_volume:
                        cand["volume_ratio"] = latest_volume / avg_volume

                    if avg_volume is not None and avg_volume < self.min_average_volume:
                        filtered_reasons["volume"] += 1
                        continue

                # Get Fundamentals and Price (fetch once, reuse in later stages)
                try:
                    from tradingagents.dataflows.y_finance import get_fundamentals, get_stock_price

                    # Get current price
                    current_price = get_stock_price(ticker)
                    cand["current_price"] = current_price

                    # Track failures for delisted cache
                    if current_price is None:
                        delisted_cache.mark_failed(ticker, "no_price_data")
                        failed_tickers.append(ticker)
                        filtered_reasons["no_data"] += 1
                        continue

                    # Get fundamentals
                    fund_json = get_fundamentals(ticker)
                    if fund_json and not fund_json.startswith("Error"):
                        fund = json.loads(fund_json)
                        cand["fundamentals"] = fund

                        # Market cap filter (if configured)
                        if self.min_market_cap:
                            market_cap_raw = fund.get("MarketCapitalization")
                            market_cap_bil = _parse_market_cap_to_billions(market_cap_raw)
                            cand["market_cap_bil"] = market_cap_bil
                            if market_cap_bil is not None and market_cap_bil < self.min_market_cap:
                                filtered_reasons["market_cap"] += 1
                                continue

                        # Extract business description for ranker LLM context
                        business_description = fund.get("Description", "")
                        if business_description and business_description != "N/A":
                            cand["business_description"] = business_description
                        else:
                            # Fallback to sector/industry description
                            sector = fund.get("Sector", "")
                            industry = fund.get("Industry", "")
                            company_name = fund.get("Name", ticker)
                            if sector and industry:
                                cand["business_description"] = (
                                    f"{company_name} is a {industry} company in the {sector} sector."
                                )
                            else:
                                cand["business_description"] = (
                                    f"{company_name} - Business description not available."
                                )
                    else:
                        cand["fundamentals"] = {}
                        cand["business_description"] = (
                            f"{ticker} - Business description not available."
                        )
                except Exception as e:
                    print(f"   Warning: Could not fetch fundamentals for {ticker}: {e}")
                    delisted_cache.mark_failed(ticker, str(e))
                    failed_tickers.append(ticker)
                    cand["current_price"] = None
                    cand["fundamentals"] = {}
                    cand["business_description"] = f"{ticker} - Business description not available."
                    filtered_reasons["no_data"] += 1
                    continue

                # Assign strategy based on source (prioritize leading indicators)
                self._assign_strategy(cand)

                # Technical Analysis Check (New)
                today_str = end_date
                rsi_data = self._run_tool(
                    state=state,
                    step="Get technical indicators",
                    tool_name="get_indicators",
                    default=None,
                    symbol=ticker,
                    curr_date=today_str,
                )
                if rsi_data:
                    cand["technical_indicators"] = rsi_data

                    # Volatility compression detection (low ATR + tight Bollinger bands)
                    atr_pct = _extract_atr_pct(rsi_data)
                    bb_width = _extract_bb_width_pct(rsi_data)
                    volume_ratio = cand.get("volume_ratio")

                    cand["atr_pct"] = atr_pct
                    cand["bb_width_pct"] = bb_width
                    has_compression = (
                        atr_pct is not None
                        and bb_width is not None
                        and atr_pct <= self.compression_atr_pct_max
                        and bb_width <= self.compression_bb_width_max
                    )
                    has_volume_uptick = (
                        volume_ratio is not None
                        and volume_ratio >= self.compression_min_volume_ratio
                    )

                    if has_compression:
                        cand["has_volatility_compression"] = has_volume_uptick
                        if has_volume_uptick:
                            compression_context = (
                                f"🧊 Volatility compression: ATR {atr_pct:.1f}%, "
                                f"BB width {bb_width:.1f}%, Vol ratio {volume_ratio:.2f}x"
                            )
                        else:
                            compression_context = (
                                f"🧊 Volatility compression: ATR {atr_pct:.1f}%, "
                                f"BB width {bb_width:.1f}%"
                            )
                        existing_context = cand.get("context", "")
                        cand["context"] = f"{existing_context} | {compression_context}"

                        if has_volume_uptick and cand.get("priority") in {"low", "medium"}:
                            cand["priority"] = "high"

                # === Per-ticker enrichment ===

                # 1. News - Use discovery news if batch news is empty/missing
                batch_news = news_by_ticker.get(ticker.upper(), news_by_ticker.get(ticker, ""))
                discovery_news = cand.get("news_context", [])

                # Prefer batch news, but fall back to discovery news if batch is empty
                if batch_news and batch_news.strip() and "No news found" not in batch_news:
                    cand["news"] = batch_news
                elif discovery_news:
                    # Convert discovery news_context to list format
                    cand["news"] = discovery_news
                else:
                    cand["news"] = ""

                # 2. Insider Transactions
                insider = self._run_tool(
                    state=state,
                    step="Get insider transactions",
                    tool_name="get_insider_transactions",
                    default="",
                    ticker=ticker,
                )
                cand["insider_transactions"] = insider or ""

                # 3. Analyst Recommendations
                recommendations = self._run_tool(
                    state=state,
                    step="Get recommendations",
                    tool_name="get_recommendation_trends",
                    default="",
                    ticker=ticker,
                )
                cand["recommendations"] = recommendations or ""

                # 4. Options Activity with Flow Analysis
                options = self._run_tool(
                    state=state,
                    step="Get options activity",
                    tool_name="get_options_activity",
                    default=None,
                    ticker=ticker,
                    num_expirations=3,
                    curr_date=end_date,
                )
                if options is None:
                    cand["options_activity"] = ""
                    cand["options_flow"] = {}
                    cand["has_bullish_options_flow"] = False
                else:
                    cand["options_activity"] = options

                    # Analyze options flow for unusual activity signals
                    from tradingagents.dataflows.y_finance import analyze_options_flow

                    options_analysis = self._run_call(
                        "analyzing options flow",
                        analyze_options_flow,
                        default={},
                        ticker=ticker,
                        num_expirations=3,
                    )
                    cand["options_flow"] = options_analysis or {}

                    # Flag unusual bullish flow as a positive signal
                    if options_analysis.get("is_bullish_flow"):
                        cand["has_bullish_options_flow"] = True
                        flow_context = (
                            f"🎯 Unusual bullish options flow: "
                            f"{options_analysis['unusual_calls']} unusual calls vs "
                            f"{options_analysis['unusual_puts']} puts, "
                            f"P/C ratio: {options_analysis['pc_volume_ratio']}"
                        )
                        # Append to context
                        existing_context = cand.get("context", "")
                        cand["context"] = f"{existing_context} | {flow_context}"
                    elif options_analysis.get("signal") in ["very_bullish", "bullish"]:
                        cand["has_bullish_options_flow"] = True
                    else:
                        cand["has_bullish_options_flow"] = False

                filtered_candidates.append(cand)

            except Exception as e:
                print(f"   Error checking {ticker}: {e}")

        return filtered_candidates, filtered_reasons, failed_tickers, delisted_cache

    def _print_filter_summary(
        self,
        candidates: List[Dict[str, Any]],
        filtered_candidates: List[Dict[str, Any]],
        filtered_reasons: Dict[str, int],
    ) -> None:
        print("\n   📊 Filtering Summary:")
        print(f"      Starting candidates: {len(candidates)}")
        if filtered_reasons.get("intraday_moved", 0) > 0:
            print(f"      ❌ Same-day movers: {filtered_reasons['intraday_moved']}")
        if filtered_reasons.get("recent_moved", 0) > 0:
            print(f"      ❌ Recent movers: {filtered_reasons['recent_moved']}")
        if filtered_reasons.get("volume", 0) > 0:
            print(f"      ❌ Low volume: {filtered_reasons['volume']}")
        if filtered_reasons.get("market_cap", 0) > 0:
            print(f"      ❌ Below market cap: {filtered_reasons['market_cap']}")
        if filtered_reasons.get("no_data", 0) > 0:
            print(f"      ❌ No data available: {filtered_reasons['no_data']}")
        print(f"      ✅ Passed filters: {len(filtered_candidates)}")

    def _run_tool(
        self,
        state: Dict[str, Any],
        step: str,
        tool_name: str,
        default: Any = None,
        **params: Any,
    ) -> Any:
        try:
            return self.execute_tool(
                state,
                node="filter",
                step=step,
                tool_name=tool_name,
                **params,
            )
        except Exception as e:
            print(f"   Error during {step}: {e}")
            return default

    def _run_call(
        self,
        label: str,
        func: Callable,
        default: Any = None,
        **kwargs: Any,
    ) -> Any:
        try:
            return func(**kwargs)
        except Exception as e:
            print(f"   Error {label}: {e}")
            return default

    def _assign_strategy(self, cand: Dict[str, Any]):
        """Assign strategy based on source."""
        source = cand.get("source", "")
        strategy = Strategy.MOMENTUM.value
        if source == "reddit_dd_undiscovered":
            strategy = Strategy.UNDISCOVERED_DD.value  # LEADING - quality research before hype
        elif source == "earnings_accumulation":
            strategy = Strategy.PRE_EARNINGS_ACCUMULATION.value  # LEADING - highest priority
        elif source == "unusual_volume":
            strategy = Strategy.EARLY_ACCUMULATION.value  # LEADING
        elif source == "analyst_upgrade":
            strategy = Strategy.ANALYST_UPGRADE.value  # LEADING - institutional signal
        elif source == "short_squeeze":
            strategy = Strategy.SHORT_SQUEEZE.value  # Event-driven - high volatility
        elif source == "semantic_news_match":
            strategy = Strategy.NEWS_CATALYST.value  # LEADING - news-driven
        elif source == "earnings_catalyst":
            strategy = Strategy.EARNINGS_PLAY.value  # Event-driven
        elif source == "ipo_listing":
            strategy = Strategy.IPO_OPPORTUNITY.value  # Event-driven
        elif source == "loser":
            strategy = Strategy.CONTRARIAN_VALUE.value
        elif source == "gainer":
            strategy = Strategy.MOMENTUM_CHASE.value
        elif source == "social_trending" or source == "twitter_sentiment":
            strategy = Strategy.SOCIAL_HYPE.value  # LAGGING
        elif source == "market_mover":
            strategy = Strategy.MOMENTUM_CHASE.value  # LAGGING - lowest priority
        cand["strategy"] = strategy

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage

from tradingagents.dataflows.discovery.utils import (
    Priority,
    append_llm_log,
    is_valid_ticker,
    resolve_llm_name,
    resolve_trade_date,
    resolve_trade_date_str,
)
from tradingagents.schemas import RedditTickerList


@dataclass
class ScannerSpec:
    name: str
    handler: Callable[[Dict[str, Any]], List[Dict[str, Any]]]
    default_priority: str = Priority.UNKNOWN.value
    enabled_key: Optional[str] = None


class TraditionalScanner:
    """
    Handles traditional market scanning strategies (Reddit, technicals, earnings, market moves).
    """

    def __init__(self, config: Dict[str, Any], llm: Any, tool_executor: Callable):
        """
        Initialize the scanner.

        Args:
            config: Configuration dictionary
            llm: Quick thinking LLM for extracting tickers from text
            tool_executor: Callback function to execute tools with logging
        """
        self.config = config
        self.llm = llm
        self.execute_tool = tool_executor

        # Extract limits
        discovery_config = config.get("discovery", {})
        self.discovery_config = discovery_config
        self.reddit_trending_limit = discovery_config.get("reddit_trending_limit", 15)
        self.market_movers_limit = discovery_config.get("market_movers_limit", 10)
        self.max_earnings_candidates = discovery_config.get("max_earnings_candidates", 50)
        self.max_days_until_earnings = discovery_config.get("max_days_until_earnings", 7)
        self.min_market_cap = discovery_config.get("min_market_cap", 0)
        self.scanner_registry = self._build_scanner_registry()

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run all traditional scanner sources and return candidates."""
        candidates: List[Dict[str, Any]] = []
        for spec in self.scanner_registry:
            if not self._scanner_enabled(spec):
                continue
            results = self._safe_scan(spec, state)
            if not results:
                continue
            for item in results:
                if not item.get("priority"):
                    item["priority"] = spec.default_priority
                if not item.get("source"):
                    item["source"] = spec.name
            candidates.extend(results)

        return self._batch_validate(state, candidates)

    def _build_scanner_registry(self) -> List[ScannerSpec]:
        return [
            ScannerSpec(
                name="reddit",
                handler=self._scan_reddit,
                default_priority=Priority.LOW.value,
                enabled_key="enable_scanner_reddit",
            ),
            ScannerSpec(
                name="market_movers",
                handler=self._scan_market_movers,
                default_priority=Priority.LOW.value,
                enabled_key="enable_scanner_market_movers",
            ),
            ScannerSpec(
                name="earnings",
                handler=self._scan_earnings,
                default_priority=Priority.MEDIUM.value,
                enabled_key="enable_scanner_earnings",
            ),
            ScannerSpec(
                name="ipo",
                handler=self._scan_ipo,
                default_priority=Priority.MEDIUM.value,
                enabled_key="enable_scanner_ipo",
            ),
            ScannerSpec(
                name="short_interest",
                handler=self._scan_short_interest,
                default_priority=Priority.MEDIUM.value,
                enabled_key="enable_scanner_short_interest",
            ),
            ScannerSpec(
                name="unusual_volume",
                handler=self._scan_unusual_volume,
                default_priority=Priority.HIGH.value,
                enabled_key="enable_scanner_unusual_volume",
            ),
            ScannerSpec(
                name="analyst_ratings",
                handler=self._scan_analyst_ratings,
                default_priority=Priority.MEDIUM.value,
                enabled_key="enable_scanner_analyst_ratings",
            ),
            ScannerSpec(
                name="insider_buying",
                handler=self._scan_insider_buying,
                default_priority=Priority.HIGH.value,
                enabled_key="enable_scanner_insider_buying",
            ),
        ]

    def _scanner_enabled(self, spec: ScannerSpec) -> bool:
        if not spec.enabled_key:
            return True
        return bool(self.discovery_config.get(spec.enabled_key, True))

    def _safe_scan(self, spec: ScannerSpec, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return spec.handler(state)
        except Exception as e:
            print(f"   Error running scanner '{spec.name}': {e}")
            return []

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
                node="scanner",
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

    def _scan_reddit(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch Reddit sources and extract tickers in a single LLM pass."""
        candidates: List[Dict[str, Any]] = []
        reddit_trending_report = None
        reddit_dd_report = None

        # 1a. Get Reddit Trending (Social Sentiment)
        reddit_trending_report = self._run_tool(
            state,
            step="Get Reddit trending tickers",
            tool_name="get_trending_tickers",
            limit=self.reddit_trending_limit,
        )

        # 1b. Get Undiscovered Reddit DD (LEADING INDICATOR)
        try:
            from tradingagents.dataflows.reddit_api import get_reddit_undiscovered_dd

            print("   🔍 Scanning Reddit for undiscovered DD...")
            # Note: get_reddit_undiscovered_dd is not a tool in strict sense but a direct function call
            # that uses an LLM. We call it directly here as in original code.
            reddit_dd_report = self._run_call(
                "fetching undiscovered DD",
                get_reddit_undiscovered_dd,
                lookback_hours=24,
                scan_limit=100,
                top_n=15,
                llm_evaluator=self.llm,  # Use fast LLM for evaluation
            )
        except Exception as e:
            print(f"   Error fetching undiscovered DD: {e}")

        # BATCHED LLM CALL: Extract tickers from both Reddit sources in ONE call
        # Uses proper Pydantic structured output for clean, validated results
        if reddit_trending_report or reddit_dd_report:
            try:
                combined_prompt = """Extract stock tickers from these Reddit reports.

IMPORTANT RULES:
1. Only extract valid US stock tickers (1-5 uppercase letters, e.g., AAPL, NVDA, TSLA)
2. Do NOT include crypto (BTC, ETH), indices (SPY, QQQ), or gibberish
3. Classify each as 'trending' (social mentions) or 'dd' (due diligence research)
4. Set confidence to 'low' if you're unsure it's a real stock ticker

"""
                if reddit_trending_report:
                    combined_prompt += f"""=== REDDIT TRENDING TICKERS ===
{reddit_trending_report}

"""
                if reddit_dd_report:
                    combined_prompt += f"""=== REDDIT UNDISCOVERED DD ===
{reddit_dd_report}

"""
                combined_prompt += """Extract ALL mentioned stock tickers with their source and context."""

                # Use proper Pydantic structured output (not raw JSON schema)
                structured_llm = self.llm.with_structured_output(RedditTickerList)
                response: RedditTickerList = structured_llm.invoke(
                    [HumanMessage(content=combined_prompt)]
                )

                tool_logs = state.get("tool_logs", [])
                append_llm_log(
                    tool_logs,
                    node="scanner",
                    step="Extract Reddit tickers",
                    model=resolve_llm_name(self.llm),
                    prompt=combined_prompt,
                    output=response.model_dump() if hasattr(response, "model_dump") else response,
                )
                state["tool_logs"] = tool_logs

                trending_count = 0
                dd_count = 0
                skipped_low_confidence = 0

                for extracted in response.tickers:
                    ticker = extracted.ticker.upper().strip()
                    source_type = extracted.source
                    context = extracted.context
                    confidence = extracted.confidence

                    # Skip low-confidence extractions (likely gibberish or crypto)
                    if confidence == "low":
                        skipped_low_confidence += 1
                        continue

                    if is_valid_ticker(ticker):
                        if source_type == "dd":
                            candidates.append(
                                {
                                    "ticker": ticker,
                                    "source": "reddit_dd_undiscovered",
                                    "context": f"💎 Undiscovered DD: {context}",
                                    "priority": "high",  # LEADING - quality DD before hype
                                }
                            )
                            dd_count += 1
                        else:
                            candidates.append(
                                {
                                    "ticker": ticker,
                                    "source": "social_trending",
                                    "context": context,
                                    "priority": "low",  # LAGGING - already trending
                                }
                            )
                            trending_count += 1

                print(
                    f"   Found {trending_count} trending + {dd_count} DD tickers from Reddit "
                    f"(skipped {skipped_low_confidence} low-confidence)"
                )
            except Exception as e:
                tool_logs = state.get("tool_logs", [])
                append_llm_log(
                    tool_logs,
                    node="scanner",
                    step="Extract Reddit tickers",
                    model=resolve_llm_name(self.llm),
                    prompt=combined_prompt,
                    output="",
                    error=str(e),
                )
                state["tool_logs"] = tool_logs
                print(f"   Error extracting Reddit tickers: {e}")

        return candidates

    def _scan_market_movers(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch top gainers and losers."""
        candidates: List[Dict[str, Any]] = []
        from tradingagents.dataflows.alpha_vantage_stock import get_top_gainers_losers

        print("   📊 Fetching market movers (direct parsing)...")
        movers_data = self._run_call(
            "fetching market movers",
            get_top_gainers_losers,
            limit=self.market_movers_limit,
            return_structured=True,
        )

        if isinstance(movers_data, dict) and not movers_data.get("error"):
            movers_count = 0
            # Process gainers
            for item in movers_data.get("gainers", []):
                ticker_raw = item.get("ticker") or ""
                ticker = ticker_raw.upper().strip() if ticker_raw else ""
                if is_valid_ticker(ticker):
                    change_pct = item.get("change_percentage") or "N/A"
                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": "gainer",
                            "context": f"Top gainer: {change_pct} change",
                            "priority": "low",  # LAGGING - already moved
                        }
                    )
                    movers_count += 1

            # Process losers
            for item in movers_data.get("losers", []):
                ticker_raw = item.get("ticker") or ""
                ticker = ticker_raw.upper().strip() if ticker_raw else ""
                if is_valid_ticker(ticker):
                    change_pct = item.get("change_percentage") or "N/A"
                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": "loser",
                            "context": f"Top loser: {change_pct} change",
                            "priority": "medium",  # Potential bounce play
                        }
                    )
                    movers_count += 1

            print(f"   Found {movers_count} market movers (direct)")
        else:
            print("   Market movers returned error or empty")

        return candidates

    def _scan_earnings(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch earnings calendar and pre-earnings accumulation signals."""
        candidates: List[Dict[str, Any]] = []
        from datetime import timedelta

        from tradingagents.dataflows.finnhub_api import get_earnings_calendar
        from tradingagents.dataflows.y_finance import get_pre_earnings_accumulation_signal

        today = resolve_trade_date(state)
        from_date = today.strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=self.max_days_until_earnings)).strftime("%Y-%m-%d")

        print(f"   📅 Fetching earnings calendar (next {self.max_days_until_earnings} days)...")
        earnings_data = self._run_call(
            "fetching earnings calendar",
            get_earnings_calendar,
            from_date=from_date,
            to_date=to_date,
            return_structured=True,
        )

        if isinstance(earnings_data, list):
            # First pass: collect all candidates with metadata
            earnings_candidates = []

            for entry in earnings_data:
                symbol = entry.get("symbol") or ""
                ticker = symbol.upper().strip() if symbol else ""
                if not is_valid_ticker(ticker):
                    continue

                # Calculate days until earnings
                earnings_date_str = entry.get("date")
                days_until = None
                if earnings_date_str:
                    try:
                        earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
                        days_until = (earnings_date - today).days
                    except Exception:
                        pass

                # Build context from structured data
                eps_est = entry.get("epsEstimate")
                date = earnings_date_str or "upcoming"
                hour = entry.get("hour") or ""
                context = f"Earnings {date}"
                if hour:
                    context += f" ({hour})"
                if eps_est is not None:
                    context += (
                        f", EPS est: ${eps_est:.2f}"
                        if isinstance(eps_est, (int, float))
                        else f", EPS est: {eps_est}"
                    )

                # Check for pre-earnings accumulation (LEADING indicator)
                has_accumulation = False
                accumulation_data = None
                accumulation = self._run_call(
                    "checking pre-earnings accumulation",
                    get_pre_earnings_accumulation_signal,
                    ticker=ticker,
                    lookback_days=10,
                )
                if isinstance(accumulation, dict) and accumulation.get("signal"):
                    has_accumulation = True
                    accumulation_data = accumulation

                earnings_candidates.append(
                    {
                        "ticker": ticker,
                        "context": context,
                        "days_until": days_until if days_until is not None else 999,
                        "has_accumulation": has_accumulation,
                        "accumulation_data": accumulation_data,
                    }
                )

            # Sort by priority: accumulation first, then by proximity to earnings
            earnings_candidates.sort(
                key=lambda x: (
                    0 if x["has_accumulation"] else 1,  # Accumulation first
                    x["days_until"],  # Then by proximity
                )
            )

            # Apply hard cap
            earnings_candidates = earnings_candidates[: self.max_earnings_candidates]

            # Add to main candidates list
            for ec in earnings_candidates:
                if ec["has_accumulation"]:
                    enhanced_context = (
                        f"{ec['context']} | "
                        f"🔥 PRE-EARNINGS ACCUMULATION: "
                        f"Vol {ec['accumulation_data']['volume_ratio']}x avg, "
                        f"Price {ec['accumulation_data']['price_change_pct']:+.1f}%"
                    )
                    candidates.append(
                        {
                            "ticker": ec["ticker"],
                            "source": "earnings_accumulation",
                            "context": enhanced_context,
                            "priority": "high",
                        }
                    )
                else:
                    candidates.append(
                        {
                            "ticker": ec["ticker"],
                            "source": "earnings_catalyst",
                            "context": ec["context"],
                            "priority": "medium",
                        }
                    )

            print(
                f"   Found {len(earnings_candidates)} earnings candidates (filtered from {len(earnings_data)} total, cap: {self.max_earnings_candidates})"
            )

        return candidates

    def _scan_ipo(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch IPO calendar."""
        candidates: List[Dict[str, Any]] = []
        from datetime import datetime, timedelta

        from tradingagents.dataflows.finnhub_api import get_ipo_calendar

        today = resolve_trade_date(state)
        from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=14)).strftime("%Y-%m-%d")

        print("   🆕 Fetching IPO calendar (direct parsing)...")
        ipo_data = self._run_call(
            "fetching IPO calendar",
            get_ipo_calendar,
            from_date=from_date,
            to_date=to_date,
            return_structured=True,
        )

        if isinstance(ipo_data, list):
            ipo_count = 0
            for entry in ipo_data:
                symbol = entry.get("symbol") or ""
                ticker = symbol.upper().strip() if symbol else ""
                if ticker and is_valid_ticker(ticker):
                    name = entry.get("name") or ""
                    date = entry.get("date", "upcoming")
                    price = entry.get("price")
                    context = f"IPO {date}: {name}"
                    if price:
                        context += f" @ ${price}"

                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": "ipo_listing",
                            "context": context,
                            "priority": "medium",
                            "allow_invalid": True,  # IPOs may not be listed yet
                        }
                    )
                    ipo_count += 1

            print(f"   Found {ipo_count} IPO candidates (direct)")

        return candidates

    def _scan_short_interest(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch short interest for squeeze candidates."""
        candidates: List[Dict[str, Any]] = []
        from tradingagents.dataflows.finviz_scraper import get_short_interest

        print("   🩳 Fetching short interest (direct parsing)...")
        short_data = self._run_call(
            "fetching short interest",
            get_short_interest,
            min_short_interest_pct=15.0,
            min_days_to_cover=5.0,
            top_n=15,
            return_structured=True,
        )

        if isinstance(short_data, list):
            short_count = 0
            for entry in short_data:
                ticker_raw = entry.get("ticker") or ""
                ticker = ticker_raw.upper().strip() if ticker_raw else ""
                if is_valid_ticker(ticker):
                    short_pct = entry.get("short_interest_pct") or 0
                    signal = entry.get("signal") or "squeeze_potential"
                    context = f"Short interest: {short_pct:.1f}%, Signal: {signal}"

                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": "short_squeeze",
                            "context": context,
                            "priority": "medium",
                        }
                    )
                    short_count += 1

            print(f"   Found {short_count} short squeeze candidates (direct)")

        return candidates

    def _scan_unusual_volume(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch unusual volume (accumulation signal)."""
        candidates: List[Dict[str, Any]] = []
        from tradingagents.dataflows.alpha_vantage_volume import get_unusual_volume

        today = resolve_trade_date_str(state)

        print("   📈 Fetching unusual volume (direct parsing)...")
        volume_data = self._run_call(
            "fetching unusual volume",
            get_unusual_volume,
            date=today,
            min_volume_multiple=2.0,
            max_price_change=5.0,
            top_n=15,
            max_tickers_to_scan=3000,
            use_cache=True,
            return_structured=True,
        )

        if isinstance(volume_data, list):
            volume_count = 0
            for entry in volume_data:
                ticker_raw = entry.get("ticker") or ""
                ticker = ticker_raw.upper().strip() if ticker_raw else ""
                if is_valid_ticker(ticker):
                    vol_ratio = entry.get("volume_ratio") or 0
                    price_change = entry.get("price_change_pct") or 0
                    intraday_change = entry.get("intraday_change_pct") or 0
                    direction = entry.get("direction") or "neutral"
                    signal = entry.get("signal") or "accumulation"

                    # Build context with direction info
                    direction_emoji = "🟢" if direction == "bullish" else "⚪"
                    context = f"Volume: {vol_ratio}x avg, Price: {price_change:+.1f}%, "
                    context += f"Intraday: {intraday_change:+.1f}% {direction_emoji}, Signal: {signal}"

                    # Strong accumulation gets highest priority
                    priority = "critical" if signal == "strong_accumulation" else "high"

                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": "unusual_volume",
                            "context": context,
                            "priority": priority,  # LEADING INDICATOR
                        }
                    )
                    volume_count += 1

            print(f"   Found {volume_count} unusual volume candidates (direct, distribution filtered)")

        return candidates

    def _scan_analyst_ratings(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch analyst rating changes."""
        candidates: List[Dict[str, Any]] = []
        from tradingagents.dataflows.alpha_vantage_analysts import get_analyst_rating_changes
        from tradingagents.dataflows.y_finance import check_if_price_reacted

        print("   📊 Fetching analyst rating changes (direct parsing)...")
        analyst_data = self._run_call(
            "fetching analyst rating changes",
            get_analyst_rating_changes,
            lookback_days=7,
            change_types=["upgrade", "initiated"],
            top_n=15,
            return_structured=True,
        )

        if isinstance(analyst_data, list):
            analyst_count = 0
            for entry in analyst_data:
                ticker_raw = entry.get("ticker") or ""
                ticker = ticker_raw.upper().strip() if ticker_raw else ""
                if is_valid_ticker(ticker):
                    action = entry.get("action") or "rating_change"
                    source = entry.get("source") or "Unknown"
                    hours_old = entry.get("hours_old") or 0

                    freshness = (
                        "🔥 FRESH"
                        if hours_old < 24
                        else "🟢 Recent" if hours_old < 72 else "Older"
                    )
                    context = f"{action.upper()} from {source} ({freshness}, {hours_old}h ago)"

                    # Check if prices already reacted
                    try:
                        reaction = check_if_price_reacted(
                            ticker, lookback_days=3, reaction_threshold=10.0
                        )
                        if reaction["status"] == "leading":
                            context += (
                                f" | 💎 EARLY: Price {reaction['price_change_pct']:+.1f}%"
                            )
                            priority = "high"
                        elif reaction["status"] == "lagging":
                            context += f" | ⚠️ LATE: Already moved {reaction['price_change_pct']:+.1f}%"
                            priority = "low"
                        else:
                            priority = "medium"
                    except Exception:
                        priority = "medium"

                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": "analyst_upgrade",
                            "context": context,
                            "priority": priority,
                        }
                    )
                    analyst_count += 1

            print(f"   Found {analyst_count} analyst upgrade candidates (direct)")

        return candidates

    def _scan_insider_buying(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch insider buying screen."""
        candidates: List[Dict[str, Any]] = []
        from tradingagents.dataflows.finviz_scraper import get_insider_buying_screener

        print("   💰 Fetching insider buying (direct parsing)...")
        insider_data = self._run_call(
            "fetching insider buying",
            get_insider_buying_screener,
            transaction_type="buy",
            lookback_days=2,
            min_value=50000,
            top_n=15,
            return_structured=True,
        )

        if isinstance(insider_data, list):
            insider_count = 0
            for entry in insider_data:
                ticker_raw = entry.get("ticker") or ""
                ticker = ticker_raw.upper().strip() if ticker_raw else ""
                if is_valid_ticker(ticker):
                    company = (entry.get("company") or "")[:30]
                    insider = (entry.get("insider") or "")[:20]
                    title = entry.get("title") or ""
                    value = entry.get("value_str") or ""

                    context = f"💰 Insider Buying: {insider} ({title}) bought {value}"
                    if company:
                        context = f"{company} - {context}"

                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": "insider_buying",
                            "context": context,
                            "priority": "high",  # LEADING - insiders know before market
                        }
                    )
                    insider_count += 1

            print(f"   Found {insider_count} insider buying candidates (direct)")

        return candidates

    def _batch_validate(
        self, state: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch validate tickers (keep IPOs even if not yet listed)."""
        if not candidates:
            return candidates

        try:
            validation = self.execute_tool(
                state,
                node="scanner",
                step="Batch validate tickers",
                tool_name="validate_tickers_batch",
                symbols=list({c.get("ticker", "") for c in candidates}),
            )
            if isinstance(validation, dict) and not validation.get("error"):
                valid_set = {t.upper() for t in validation.get("valid", [])}
                invalid_list = validation.get("invalid", [])
                if valid_set or len(invalid_list) < len(candidates):
                    before_count = len(candidates)
                    candidates = [
                        c
                        for c in candidates
                        if c.get("allow_invalid") or c.get("ticker", "").upper() in valid_set
                    ]
                    removed = before_count - len(candidates)
                    if removed:
                        print(f"   Removed {removed} invalid tickers after batch validation.")
                else:
                    print("   Batch validation returned no valid tickers; skipping filter.")
        except Exception as e:
            print(f"   Error during batch validation: {e}")

        return candidates

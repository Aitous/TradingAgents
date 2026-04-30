"""Earnings calendar scanner for upcoming earnings events."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.tools.executor import execute_tool
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class EarningsCalendarScanner(BaseScanner):
    """Scan for stocks with upcoming earnings (volatility plays)."""

    name = "earnings_calendar"
    pipeline = "events"
    strategy = "earnings_play"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_candidates = self.scanner_config.get("max_candidates", 25)
        self.max_days_until_earnings = self.scanner_config.get("max_days_until_earnings", 3)
        self.min_market_cap = self.scanner_config.get("min_market_cap", 0)
        # Quality filter config
        self.iv_min_expansion = self.scanner_config.get("iv_min_expansion", 0.20)
        self.si_max_surprise_pct = self.scanner_config.get("si_max_surprise_pct", 10.0)
        self.require_iv_expansion = self.scanner_config.get("require_iv_expansion", False)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info(f"📅 Scanning earnings calendar (next {self.max_days_until_earnings} days)...")

        try:
            # Get earnings calendar from Finnhub or Alpha Vantage
            from_date = datetime.now().strftime("%Y-%m-%d")
            to_date = (datetime.now() + timedelta(days=self.max_days_until_earnings)).strftime(
                "%Y-%m-%d"
            )

            cal_result = execute_tool("get_earnings_calendar", from_date=from_date, to_date=to_date)

            if not cal_result:
                logger.info("Found 0 earnings events")
                return []

            candidates = []
            seen_tickers = set()

            # Parse earnings data
            if isinstance(cal_result, list):
                # Structured list of earnings
                candidates = self._parse_structured_earnings(cal_result, seen_tickers)
            elif isinstance(cal_result, dict):
                # Dict format
                earnings_list = cal_result.get("earnings", cal_result.get("data", []))
                candidates = self._parse_structured_earnings(earnings_list, seen_tickers)
            elif isinstance(cal_result, str):
                # Text/markdown format
                candidates = self._parse_text_earnings(cal_result, seen_tickers)

            # Sort by days until earnings (sooner = higher priority)
            candidates.sort(key=lambda x: x.get("days_until", 999))

            # Build insider_buying ticker set once for the whole batch (past 3 days)
            insider_tickers = self._load_recent_insider_tickers(days_back=3)

            # Enrich top candidates with accumulation signal and EPS estimates.
            # Only enrich within the tighter 2–3 day window; beyond that the
            # setup hasn't matured and enrichment adds noise, not signal.
            enriched = []
            for cand in candidates[:10]:
                days_until = cand.get("days_until", 999)
                if 2 <= days_until <= 3:
                    enrich_result = self._enrich_earnings_candidate(cand, insider_tickers)
                    if enrich_result is None:
                        # Hard rejection (e.g. SI surprise) — drop this candidate
                        continue
                    # Post-enrichment gate: without accumulation confirmation a
                    # pure event-day binary play is low-conviction.  Cap priority
                    # at 70 so it never reaches CRITICAL without confirmation.
                    if not cand.get("has_accumulation"):
                        cand["priority"] = min(cand.get("priority", 70), 70)
                enriched.append(cand)
            candidates = enriched + candidates[10:]

            # Apply limit
            candidates = candidates[: self.limit]

            logger.info(f"Found {len(candidates)} upcoming earnings")
            return candidates

        except Exception as e:
            logger.warning(f"⚠️  Earnings calendar failed: {e}")
            return []

    def _enrich_earnings_candidate(
        self, cand: Dict[str, Any], insider_tickers: set
    ) -> Optional[Dict[str, Any]]:
        """Enrich earnings candidate with quality filters and estimates (in-place).

        Applies three quality filters before standard enrichment:
        1. Short interest surprise (hard rejection — returns None).
        2. IV expansion (score boost or soft rejection).
        3. Insider buying conflict (score cap at 65).

        Returns the mutated candidate dict, or None if hard-rejected.
        """
        ticker = cand["ticker"]

        # ── Filter 1: Short Interest Surprise (hard rejection) ─────────────────
        si_surprise = self._check_si_surprise(ticker)
        if si_surprise:
            logger.info(
                f"earnings_calendar: {ticker} rejected — SI surprise "
                f"(+{si_surprise:.1f}% above trend)"
            )
            return None

        # ── Filter 2: IV Expansion (score boost) ───────────────────────────────
        iv_expansion_pct = self._check_iv_expansion(ticker)
        if iv_expansion_pct is not None and iv_expansion_pct >= self.iv_min_expansion:
            cand["iv_expansion_pct"] = round(iv_expansion_pct * 100, 1)
            cand["context"] += f" | IV expanded {cand['iv_expansion_pct']:.0f}% (2d)"
            cand["priority"] = min(cand.get("priority", 0) + 10, Priority.CRITICAL.value)
            logger.debug(
                f"earnings_calendar: {ticker} IV expanded "
                f"{cand['iv_expansion_pct']:.0f}% — +10 priority pts"
            )
        elif self.require_iv_expansion and iv_expansion_pct is not None:
            # require_iv_expansion=True and expansion did NOT meet threshold
            logger.info(
                f"earnings_calendar: {ticker} rejected — IV expansion "
                f"{(iv_expansion_pct or 0)*100:.0f}% < required {self.iv_min_expansion*100:.0f}%"
            )
            return None

        # ── Filter 3: Insider Buying Conflict (score cap) ─────────────────────
        if ticker in insider_tickers:
            old_priority = cand.get("priority", 0)
            cand["priority"] = min(old_priority, 65)
            cand["insider_buying_conflict"] = True
            cand["context"] += " | insider_buying signal in past 3d — priority capped at 65"
            logger.debug(
                f"earnings_calendar: {ticker} insider conflict — "
                f"priority capped 65 (was {old_priority})"
            )

        # ── Standard enrichment ────────────────────────────────────────────────
        # Check pre-earnings volume accumulation
        try:
            from tradingagents.dataflows.y_finance import get_pre_earnings_accumulation_signal

            signal = get_pre_earnings_accumulation_signal(ticker)
            if signal and signal.get("signal"):
                vol_ratio = signal.get("volume_ratio", 0)
                cand["has_accumulation"] = True
                cand["accumulation_volume_ratio"] = vol_ratio
                cand["context"] += f" | Pre-earnings accumulation: {vol_ratio:.1f}x volume"
                cand["priority"] = Priority.CRITICAL.value
        except Exception:
            pass

        # Add earnings estimates
        try:
            from tradingagents.dataflows.finnhub_api import get_ticker_earnings_estimate

            est = get_ticker_earnings_estimate(ticker)
            if est and est.get("has_upcoming_earnings"):
                eps = est.get("eps_estimate")
                if eps is not None:
                    cand["eps_estimate"] = eps
                    cand["context"] += f" | EPS est: ${eps:.2f}"
        except Exception:
            pass

        return cand

    # ── Quality-filter helpers ─────────────────────────────────────────────────

    def _check_iv_expansion(self, ticker: str) -> Optional[float]:
        """Return fractional IV expansion over the past 2 days, or None if unavailable.

        Proxy: compares the current front-month ATM implied volatility against
        the stock's 10-day realized volatility (annualised).  A positive ratio
        above the configured threshold signals that options traders have been
        paying up for protection — a classic informed-money footprint.

        Returns a float (e.g. 0.35 = 35% above HV baseline), or None on error.
        """
        try:
            import math

            import yfinance as yf

            from tradingagents.dataflows.y_finance import suppress_yfinance_warnings

            with suppress_yfinance_warnings():
                tk = yf.Ticker(ticker.upper())

                # Get nearest expiration options chain
                expirations = tk.options
                if not expirations:
                    return None

                opt = tk.option_chain(expirations[0])
                calls = opt.calls
                if calls.empty or "impliedVolatility" not in calls.columns:
                    return None

                # Find ATM strike: the strike closest to current price
                hist = tk.history(period="5d")
                if hist.empty:
                    return None
                current_price = float(hist["Close"].iloc[-1])

                calls_valid = calls[calls["impliedVolatility"].notna()].copy()
                if calls_valid.empty:
                    return None

                calls_valid["strike_dist"] = (calls_valid["strike"] - current_price).abs()
                atm_row = calls_valid.nsmallest(1, "strike_dist")
                current_iv = float(atm_row["impliedVolatility"].iloc[0])  # decimal form

                # Compute 10-day realised volatility as baseline
                hist_30 = tk.history(period="30d")
                if len(hist_30) < 10:
                    return None

                closes = hist_30["Close"].dropna()
                log_returns = closes.pct_change().dropna()
                if len(log_returns) < 9:
                    return None

                hv_daily = float(log_returns.tail(10).std())
                hv_annual = hv_daily * math.sqrt(252)

                if hv_annual <= 0:
                    return None

                # Expansion ratio: (IV - HV) / HV
                expansion = (current_iv - hv_annual) / hv_annual
                return expansion

        except Exception as exc:
            logger.debug(f"earnings_calendar: IV expansion check failed for {ticker}: {exc}")
            return None

    def _check_si_surprise(self, ticker: str) -> Optional[float]:
        """Return the SI surprise percentage if it exceeds the configured cap, else None.

        Uses yfinance ticker info to get current short interest % of float and
        short ratio.  Compares against the configured threshold.  If SI has
        surged beyond the cap, returns the excess percentage (positive float).
        Returns None if data is unavailable or the threshold is not breached.

        Note: True trend-based SI surprise would require two settlement snapshots
        (bi-monthly FINRA data).  As a practical proxy we reject tickers whose
        current SI % of float exceeds a hard cap (default: 10%), indicating a
        crowded short base that creates binary gap-risk against a long position.
        """
        try:
            from tradingagents.dataflows.y_finance import get_ticker_info

            info = get_ticker_info(ticker)
            if not info:
                return None

            short_pct = info.get("shortPercentOfFloat") or info.get("sharesPercentSharesOut")
            if short_pct is None or not isinstance(short_pct, (int, float)):
                return None

            short_pct_display = float(short_pct) * 100  # convert to percentage

            if short_pct_display > self.si_max_surprise_pct:
                return short_pct_display - self.si_max_surprise_pct

            return None

        except Exception as exc:
            logger.debug(f"earnings_calendar: SI check failed for {ticker}: {exc}")
            return None

    def _load_recent_insider_tickers(self, days_back: int = 3) -> set:
        """Return set of tickers that appeared as insider_buying in past N days.

        Reads scanner_picks/YYYY-MM-DD.json files from the data directory.
        Gracefully returns an empty set if files are missing or unreadable.
        """
        insider_tickers: set = set()
        data_dir = Path(self.config.get("data_dir", "data"))
        picks_dir = data_dir / "scanner_picks"
        today = datetime.now().date()

        for days_ago in range(1, days_back + 1):
            check_date = today - timedelta(days=days_ago)
            picks_file = picks_dir / f"{check_date.isoformat()}.json"
            if not picks_file.exists():
                continue
            try:
                with open(picks_file) as fh:
                    data = json.load(fh)
                for pick in data.get("picks", []):
                    if pick.get("scanner") == "insider_buying" or pick.get(
                        "strategy"
                    ) in ("insider_buying", "insider_cluster_buying"):
                        ticker = pick.get("ticker", "").upper()
                        if ticker:
                            insider_tickers.add(ticker)
            except Exception as exc:
                logger.debug(
                    f"earnings_calendar: Could not load scanner_picks "
                    f"for {check_date}: {exc}"
                )

        if insider_tickers:
            logger.debug(
                f"earnings_calendar: {len(insider_tickers)} insider_buying "
                f"tickers found in past {days_back} days"
            )
        return insider_tickers

    def _parse_structured_earnings(
        self, earnings_list: List[Dict], seen_tickers: set
    ) -> List[Dict[str, Any]]:
        """Parse structured earnings data."""
        candidates = []
        today = datetime.now().date()

        for event in earnings_list[: self.max_candidates * 2]:
            ticker = event.get("ticker", event.get("symbol", "")).upper()
            if not ticker or ticker in seen_tickers:
                continue

            # Get earnings date
            earnings_date_str = event.get("date", event.get("earnings_date", ""))
            if not earnings_date_str:
                continue

            try:
                # Parse date (handle different formats)
                if isinstance(earnings_date_str, str):
                    earnings_date = datetime.strptime(
                        earnings_date_str.split()[0], "%Y-%m-%d"
                    ).date()
                else:
                    earnings_date = earnings_date_str

                days_until = (earnings_date - today).days

                # Filter by max days
                if days_until < 0 or days_until > self.max_days_until_earnings:
                    continue

                # Filter by market cap if specified
                market_cap = event.get("market_cap", 0)
                if self.min_market_cap > 0 and market_cap < self.min_market_cap * 1e9:
                    continue

                seen_tickers.add(ticker)

                # Priority based on proximity to earnings
                if days_until <= 2:
                    priority = Priority.HIGH.value
                elif days_until <= 5:
                    priority = Priority.MEDIUM.value
                else:
                    priority = Priority.LOW.value

                candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.name,
                        "context": f"Earnings in {days_until} day(s) on {earnings_date_str}",
                        "priority": priority,
                        "strategy": (
                            "pre_earnings_accumulation" if days_until > 1 else "earnings_play"
                        ),
                        "days_until": days_until,
                        "earnings_date": earnings_date_str,
                    }
                )

                if len(candidates) >= self.max_candidates:
                    break

            except (ValueError, AttributeError):
                continue

        return candidates

    def _parse_text_earnings(self, text: str, seen_tickers: set) -> List[Dict[str, Any]]:
        """Parse earnings from text/markdown format."""
        import re

        candidates = []
        today = datetime.now().date()

        # Split by date sections (### 2026-02-05)
        date_sections = re.split(r"###\s+(\d{4}-\d{2}-\d{2})", text)

        current_date = None
        for i, section in enumerate(date_sections):
            # Check if this is a date line
            if re.match(r"\d{4}-\d{2}-\d{2}", section):
                current_date = section
                continue

            if not current_date:
                continue

            # Find tickers in this section (format: **TICKER** (timing))
            ticker_pattern = r"\*\*([A-Z]{2,5})\*\*\s*\(([^\)]+)\)"
            ticker_matches = re.findall(ticker_pattern, section)

            for ticker, timing in ticker_matches:
                if ticker in seen_tickers:
                    continue

                try:
                    earnings_date = datetime.strptime(current_date, "%Y-%m-%d").date()
                    days_until = (earnings_date - today).days

                    if days_until < 0 or days_until > self.max_days_until_earnings:
                        continue

                    seen_tickers.add(ticker)

                    # Priority based on proximity and timing
                    if days_until <= 1:
                        priority = Priority.HIGH.value
                    elif days_until <= 3:
                        priority = Priority.MEDIUM.value
                    else:
                        priority = Priority.LOW.value

                    # Strategy based on timing
                    if timing == "bmo":  # Before market open
                        strategy = "earnings_play"
                    elif timing == "amc":  # After market close
                        strategy = (
                            "pre_earnings_accumulation" if days_until > 0 else "earnings_play"
                        )
                    else:
                        strategy = "pre_earnings_accumulation"

                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": self.name,
                            "context": f"Earnings {timing} in {days_until} day(s) on {current_date}",
                            "priority": priority,
                            "strategy": strategy,
                            "days_until": days_until,
                            "earnings_date": current_date,
                            "timing": timing,
                        }
                    )

                    if len(candidates) >= self.max_candidates:
                        return candidates

                except ValueError:
                    continue

        return candidates


SCANNER_REGISTRY.register(EarningsCalendarScanner)

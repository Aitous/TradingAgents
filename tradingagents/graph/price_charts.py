"""Price chart building and rendering for discovery recommendations.

Extracts all chart-related logic (fetching price data, rendering charts,
computing movement stats) into a standalone class so that DiscoveryGraph
stays focused on orchestration.
"""

from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class PriceChartBuilder:
    """Builds per-ticker console price charts and movement statistics."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        library: str = "plotille",
        windows: Any = None,
        lookback_days: int = 30,
        width: int = 60,
        height: int = 12,
        max_tickers: int = 10,
        show_movement_stats: bool = True,
    ) -> None:
        self.enabled = enabled
        self.library = library
        self.raw_windows = windows if windows is not None else ["1m"]
        self.lookback_days = lookback_days
        self.width = width
        self.height = height
        self.max_tickers = max_tickers
        self.show_movement_stats = show_movement_stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_bundle(self, rankings_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build per-ticker chart + movement stats for top recommendations."""
        if not self.enabled:
            return {}

        tickers = _unique_tickers(rankings_list, self.max_tickers)
        if not tickers:
            return {}

        chart_windows = self._normalize_windows()
        renderer = self._get_renderer()
        if renderer is None:
            return {}

        bundle: Dict[str, Dict[str, Any]] = {}
        for ticker in tickers:
            series = self._fetch_price_series(ticker)
            if not series:
                bundle[ticker] = {
                    "chart": f"{ticker}: no price history available",
                    "charts": {},
                    "movement": {},
                }
                continue

            per_window_charts: Dict[str, str] = {}
            for window in chart_windows:
                window_closes = self._get_window_closes(ticker, series, window)
                if len(window_closes) < 2:
                    continue

                change_pct = None
                if window_closes[0]:
                    change_pct = (window_closes[-1] / window_closes[0] - 1) * 100.0

                label = window.upper()
                title = f"{ticker} ({label})"
                if change_pct is not None:
                    title = f"{ticker} ({label}, {change_pct:+.1f}%)"

                chart_text = renderer(window_closes, title)
                if chart_text:
                    per_window_charts[window] = chart_text

            primary_chart = ""
            if per_window_charts:
                first_key = chart_windows[0]
                primary_chart = per_window_charts.get(
                    first_key, next(iter(per_window_charts.values()))
                )

            bundle[ticker] = {
                "chart": primary_chart,
                "charts": per_window_charts,
                "movement": _compute_movement_stats(series),
            }
        return bundle

    def build_map(self, rankings_list: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build mini price charts keyed by ticker."""
        bundle = self.build_bundle(rankings_list)
        return {ticker: item.get("chart", "") for ticker, item in bundle.items()}

    def build_strings(self, rankings_list: List[Dict[str, Any]]) -> List[str]:
        """Build mini price charts for top recommendations (returns ANSI strings)."""
        charts = self.build_map(rankings_list)
        return list(charts.values()) if charts else []

    def print_charts(self, rankings_list: List[Dict[str, Any]]) -> None:
        """Render mini price charts for top recommendations in the console."""
        charts = self.build_strings(rankings_list)
        if not charts:
            return

        logger.info(f"📈 Price Charts (last {self.lookback_days} days)")
        for chart in charts:
            logger.info(chart)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_price_series(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch recent daily close prices with dates for charting and movement stats."""
        try:
            from tradingagents.dataflows.y_finance import download_history

            history_days = max(self.lookback_days + 10, 390)
            data = download_history(
                ticker,
                period=f"{history_days}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )

            series = _extract_close_series(data)
            if series is None:
                return []

            points: List[Dict[str, Any]] = []
            for idx, close in series.items():
                dt = getattr(idx, "to_pydatetime", lambda: idx)()
                points.append({"date": dt, "close": float(close)})
            return points
        except Exception as exc:
            logger.error(f"{ticker}: error fetching prices: {exc}")
            return []

    def _fetch_intraday_closes(self, ticker: str) -> List[float]:
        """Fetch intraday close prices for 1-day chart window."""
        try:
            from tradingagents.dataflows.y_finance import download_history

            data = download_history(
                ticker,
                period="1d",
                interval="15m",
                auto_adjust=True,
                progress=False,
            )

            series = _extract_close_series(data)
            if series is None:
                return []

            return [float(value) for value in series.to_list()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Window / renderer helpers
    # ------------------------------------------------------------------

    def _normalize_windows(self) -> List[str]:
        """Normalize configured chart windows."""
        allowed = {"1d", "7d", "1m", "6m", "1y"}
        configured = self.raw_windows
        if isinstance(configured, str):
            configured = [part.strip().lower() for part in configured.split(",")]
        elif not isinstance(configured, list):
            configured = ["1m"]

        windows: List[str] = []
        for value in configured:
            key = str(value).strip().lower()
            if key in allowed and key not in windows:
                windows.append(key)
        return windows or ["1m"]

    def _get_window_closes(
        self, ticker: str, series: List[Dict[str, Any]], window: str
    ) -> List[float]:
        """Return closes for a given chart window."""
        if not series:
            return []

        if window == "1d":
            intraday = self._fetch_intraday_closes(ticker)
            if len(intraday) >= 2:
                return intraday
            return [point["close"] for point in series[-2:]]

        window_days = {
            "7d": 7,
            "1m": 30,
            "6m": 182,
            "1y": 365,
        }.get(window, self.lookback_days)

        latest_date = series[-1]["date"]
        cutoff = latest_date - timedelta(days=window_days)
        return [point["close"] for point in series if point["date"] >= cutoff]

    def _get_renderer(self) -> Optional[Callable[[List[float], str], str]]:
        """Return selected chart renderer, with fallback to plotext."""
        preferred = str(self.library or "plotext").lower().strip()

        if preferred == "plotille":
            try:
                import plotille

                return lambda closes, title: self._render_plotille(plotille, closes, title)
            except Exception as exc:
                logger.warning(f"⚠️  plotille unavailable, falling back to plotext: {exc}")

        try:
            import plotext as plt

            return lambda closes, title: self._render_plotext(plt, closes, title)
        except Exception as exc:
            logger.warning(f"⚠️  plotext not available, skipping charts: {exc}")
            return None

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def _render_plotille(self, plotille: Any, closes: List[float], title: str) -> str:
        """Build a plotille chart and return as ANSI string."""
        if not closes:
            return ""

        fig = plotille.Figure()
        fig.width = self.width
        fig.height = self.height
        fig.color_mode = "byte"
        fig.set_x_limits(min_=0, max_=max(1, len(closes) - 1))

        min_close = min(closes)
        max_close = max(closes)
        if min_close == max_close:
            padding = max(0.01, min_close * 0.01)
            min_close -= padding
            max_close += padding
        fig.set_y_limits(min_=min_close, max_=max_close)
        fig.plot(range(len(closes)), closes, lc=45)

        return f"{title}\n{fig.show(legend=False)}"

    def _render_plotext(self, plt: Any, closes: List[float], title: str) -> str:
        """Build a single plotext line chart and return as ANSI string."""
        _reset_plotext(plt)

        if hasattr(plt, "plotsize"):
            plt.plotsize(self.width, self.height)

        if hasattr(plt, "theme"):
            try:
                plt.theme("pro")
            except Exception:
                pass

        if hasattr(plt, "title"):
            plt.title(title)

        if hasattr(plt, "xlabel"):
            plt.xlabel("")
        if hasattr(plt, "ylabel"):
            plt.ylabel("")

        plt.plot(closes)

        if hasattr(plt, "build"):
            chart = plt.build()
            if chart:
                return chart

        plt.show()
        return ""


# ------------------------------------------------------------------
# Module-level helpers (stateless)
# ------------------------------------------------------------------


def _unique_tickers(rankings_list: List[Dict[str, Any]], limit: int) -> List[str]:
    """Extract unique uppercase tickers from a rankings list, up to *limit*."""
    tickers: List[str] = []
    for item in rankings_list:
        ticker = (item.get("ticker") or "").upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers[:limit]


def _extract_close_series(data: Any) -> Any:
    """
    Extract the Close column from a yfinance DataFrame, handling MultiIndex.

    Returns a pandas Series of close prices with NaNs dropped, or None if
    the input is empty.
    """
    import pandas as pd

    if data is None or data.empty:
        return None

    series = None
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close_data = data["Close"]
            series = (
                close_data.iloc[:, 0]
                if isinstance(close_data, pd.DataFrame)
                else close_data
            )
    elif "Close" in data.columns:
        series = data["Close"]

    if series is None:
        series = data.iloc[:, 0]

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.dropna()
    return series if not series.empty else None


def _compute_movement_stats(series: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Compute 1D, 7D, 1M, 6M, and 1Y percent movement from latest close."""
    if not series:
        return {}

    latest = series[-1]
    latest_date = latest["date"]
    latest_close = latest["close"]

    if not latest_close:
        return {}

    windows = {
        "1d": timedelta(days=1),
        "7d": timedelta(days=7),
        "1m": timedelta(days=30),
        "6m": timedelta(days=182),
        "1y": timedelta(days=365),
    }

    stats: Dict[str, Optional[float]] = {}
    for label, delta in windows.items():
        target_date = latest_date - delta
        baseline = None
        for point in series:
            if point["date"] <= target_date:
                baseline = point["close"]
            else:
                break

        if baseline and baseline != 0:
            stats[label] = (latest_close / baseline - 1.0) * 100.0
        else:
            stats[label] = None
    return stats


def _reset_plotext(plt: Any) -> None:
    """Clear plotext state between charts."""
    for method in ("clf", "clear_figure", "clear_data"):
        func = getattr(plt, method, None)
        if callable(func):
            func()
            return

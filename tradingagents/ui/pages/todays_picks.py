"""
Signals page — today's recommendation cards with rich visual indicators.

Each signal is displayed as a data-dense card with strategy badges,
confidence bars, and expandable thesis sections.
"""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tradingagents.ui.theme import COLORS, get_plotly_template, page_header, signal_card
from tradingagents.ui.utils import load_recommendations

TIMEFRAME_LOOKBACK_DAYS = {
    "1D": 1,
    "7D": 7,
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
}


def _get_interval_for_timeframe(timeframe: str) -> str:
    """Return appropriate data interval for a given timeframe."""
    if timeframe == "1D":
        return "5m"  # 5-minute for intraday detail
    elif timeframe == "7D":
        return "1h"  # Hourly for smooth 7-day view
    else:
        return "1d"  # Daily for longer timeframes


@st.cache_data(ttl=3600)
def _load_price_history(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    try:
        from tradingagents.dataflows.y_finance import download_history
    except Exception:
        return pd.DataFrame()

    data = download_history(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        tickers = data.columns.get_level_values(1).unique()
        target = ticker if ticker in tickers else tickers[0]
        data = data.xs(target, level=1, axis=1).copy()

    data = data.reset_index()
    # yfinance uses "Datetime" for intraday and "Date" for daily data
    date_col = next(
        (c for c in ("Datetime", "Date") if c in data.columns),
        data.columns[0],
    )
    close_col = "Close" if "Close" in data.columns else "Adj Close"
    if close_col not in data.columns:
        return pd.DataFrame()

    history = data[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
    history["date"] = pd.to_datetime(history["date"], utc=True).dt.tz_localize(None)
    history = history.dropna(subset=["close"]).sort_values("date")
    return history


def _slice_history_window(history: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    days = TIMEFRAME_LOOKBACK_DAYS.get(timeframe)
    if history.empty or days is None:
        return pd.DataFrame()

    latest_date = history["date"].max()
    cutoff = latest_date - pd.Timedelta(days=days)
    window = history[history["date"] >= cutoff].copy()
    if len(window) < 2:
        return pd.DataFrame()
    return window


def _format_move_pct(window: pd.DataFrame) -> str:
    first_close = float(window["close"].iloc[0])
    last_close = float(window["close"].iloc[-1])
    if first_close == 0:
        return "0.00%"
    move = ((last_close - first_close) / first_close) * 100
    return f"{move:+.2f}%"


def _get_daily_movement(ticker: str) -> str:
    """Get today's intraday price movement percentage."""
    try:
        from tradingagents.dataflows.y_finance import download_history

        today_data = download_history(
            ticker,
            period="1d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if today_data is None or today_data.empty:
            return "N/A"

        if isinstance(today_data.columns, pd.MultiIndex):
            tickers = today_data.columns.get_level_values(1).unique()
            if ticker not in tickers:
                ticker = tickers[0]
            today_data = today_data.xs(ticker, level=1, axis=1)

        close_col = "Close" if "Close" in today_data.columns else "Adj Close"
        if close_col not in today_data.columns:
            return "N/A"

        today_close = float(today_data[close_col].iloc[-1])
        today_open = float(today_data.get("Open", today_data[close_col]).iloc[-1])
        if today_open != 0:
            daily_move = ((today_close - today_open) / today_open) * 100
            return f"{daily_move:+.2f}%"
    except Exception:
        pass
    return "N/A"


def _is_intraday(timeframe: str) -> bool:
    """Return True for timeframes that use intraday (sub-daily) data."""
    return timeframe in ("1D", "7D")


def _build_dynamic_chart(
    history: pd.DataFrame, timeframe: str, ticker: str = ""
) -> tuple[go.Figure, str, str, str]:
    window = _slice_history_window(history, timeframe)
    if window.empty:
        return go.Figure(), "N/A", COLORS["text_muted"], "N/A"

    first_close = float(window["close"].iloc[0])
    last_close = float(window["close"].iloc[-1])
    line_color = COLORS["green"] if last_close >= first_close else COLORS["red"]
    move_text = _format_move_pct(window)
    daily_move_text = _get_daily_movement(ticker) if ticker else "N/A"

    intraday = _is_intraday(timeframe)

    template = dict(get_plotly_template())

    fig = go.Figure()

    if intraday:
        # For intraday charts, plot against a sequential index to eliminate
        # overnight and weekend gaps. Use customdata for hover timestamps.
        window = window.reset_index(drop=True)
        x_vals = list(range(len(window)))

        # Build tick labels: show date at the start of each trading day
        tick_vals = []
        tick_labels = []
        prev_day = None
        for i, row in window.iterrows():
            day = row["date"].date()
            if day != prev_day:
                tick_vals.append(i)
                tick_labels.append(row["date"].strftime("%b %d"))
                prev_day = day

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=window["close"],
                mode="lines",
                line=dict(color=line_color, width=2.4),
                fill="tozeroy",
                fillcolor=(
                    "rgba(34,197,94,0.18)"
                    if line_color == COLORS["green"]
                    else "rgba(239,68,68,0.18)"
                ),
                customdata=window["date"],
                hovertemplate="%{customdata|%b %d %H:%M}<br>$%{y:.2f}<extra></extra>",
                name=f"{timeframe} Focus",
            )
        )
    else:
        # For daily charts, keep the original two-trace approach
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["close"],
                mode="lines",
                line=dict(color="rgba(148,163,184,0.22)", width=1.1),
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
                name="History",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=window["date"],
                y=window["close"],
                mode="lines",
                line=dict(color=line_color, width=2.8),
                fill="tozeroy",
                fillcolor=(
                    "rgba(34,197,94,0.18)"
                    if line_color == COLORS["green"]
                    else "rgba(239,68,68,0.18)"
                ),
                hovertemplate=f"{timeframe}<br>%{{x|%b %d, %Y}}<br>$%{{y:.2f}}<extra></extra>",
                name=f"{timeframe} Focus",
            )
        )

    # Override template keys before expansion to avoid duplicate keyword args.
    template["height"] = 210
    template["showlegend"] = False
    template["margin"] = dict(l=0, r=0, t=10, b=0)
    fig.update_layout(**template)

    # Tighten Y-axis to selected timeframe range for better signal visibility.
    y_min = float(window["close"].min())
    y_max = float(window["close"].max())
    if y_min == y_max:
        pad = max(0.5, y_min * 0.01)
    else:
        pad = max((y_max - y_min) * 0.08, y_max * 0.01)

    if intraday:
        fig.update_xaxes(
            showticklabels=True,
            showgrid=False,
            tickvals=tick_vals,
            ticktext=tick_labels,
            tickfont=dict(size=9, color=COLORS["text_muted"]),
            rangeslider=dict(visible=False),
        )
    else:
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            range=[window["date"].min(), history["date"].max()],
            rangeslider=dict(visible=False),
        )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(42,53,72,0.28)",
        tickprefix="$",
        nticks=5,
        range=[y_min - pad, y_max + pad],
    )
    return fig, move_text, line_color, daily_move_text


def _render_single_dynamic_chart(ticker: str, timeframe: str) -> None:
    # Load base history with daily data for context (full year)
    base_history = _load_price_history(ticker, "1y", interval="1d")
    if base_history.empty:
        st.caption("No price history available for this ticker.")
        return

    # For 1D and 7D, load higher granularity (intraday) data
    intraday = _is_intraday(timeframe)
    if intraday:
        # Map timeframe to yfinance period: 1D -> "1d", 7D -> "7d"
        yf_period = {"1D": "1d", "7D": "7d"}[timeframe]
        interval = _get_interval_for_timeframe(timeframe)
        history_for_chart = _load_price_history(ticker, yf_period, interval=interval)

        if history_for_chart.empty or len(history_for_chart) < 2:
            # Fallback to daily data
            history_for_chart = base_history
            intraday = False

        if intraday:
            # Use all loaded intraday data directly — yfinance already
            # returned exactly the period we asked for.
            window = history_for_chart
        else:
            window = _slice_history_window(history_for_chart, timeframe)
    else:
        history_for_chart = base_history
        window = _slice_history_window(history_for_chart, timeframe)

    if window.empty or len(window) < 2:
        # Last-resort fallback: use all available data
        if len(history_for_chart) >= 2:
            window = history_for_chart.copy()
        else:
            st.caption(f"Not enough data to render {timeframe} window.")
            return

    fig, move_text, move_color, daily_move_text = _build_dynamic_chart(
        history_for_chart, timeframe, ticker
    )

    # Determine daily movement color
    try:
        daily_move_val = float(daily_move_text.strip().rstrip("%"))
        daily_color = COLORS["green"] if daily_move_val >= 0 else COLORS["red"]
    except (ValueError, AttributeError):
        daily_color = COLORS["text_muted"]

    st.markdown(
        f"""
        <div style="margin-top:0.4rem;margin-bottom:0.45rem;padding:0.45rem 0.6rem;
            border:1px solid {COLORS['border']};border-radius:8px;
            background:linear-gradient(120deg, rgba(6,182,212,0.10), rgba(59,130,246,0.05));
            display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                    text-transform:uppercase;letter-spacing:0.07em;color:{COLORS['text_secondary']};">
                    Dynamic Price View • {timeframe}
                </span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                    color:{COLORS['text_muted']};margin-left:0.8rem;">
                    Daily: <span style="color:{daily_color};font-weight:600;">{daily_move_text}</span>
                </span>
            </div>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                font-weight:700;color:{move_color};">{move_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render():
    today = datetime.now().strftime("%Y-%m-%d")
    recommendations, meta = load_recommendations(today, return_meta=True)
    display_date = meta.get("date", today) if meta else today

    st.markdown(
        page_header("Signals", f"Recommendations for {display_date}"),
        unsafe_allow_html=True,
    )

    if not recommendations:
        st.warning(f"No recommendations for {today}.")
        return

    if meta.get("is_fallback") and meta.get("date"):
        st.info(f"Showing latest signals from **{meta['date']}** (none for today).")

    # ---- Controls row ----
    ctrl_cols = st.columns([1, 1, 1, 1])
    with ctrl_cols[0]:
        pipelines = sorted(
            set(
                (r.get("strategy_match") or r.get("pipeline") or "unknown") for r in recommendations
            )
        )
        pipeline_filter = st.multiselect("Strategy", pipelines, default=pipelines)
    with ctrl_cols[1]:
        min_confidence = st.slider("Min Confidence", 1, 10, 1)
    with ctrl_cols[2]:
        min_score = st.slider("Min Score", 0, 100, 0)
    with ctrl_cols[3]:
        show_charts = st.checkbox("Price Charts", value=False)

    # Apply filters
    filtered = [
        r
        for r in recommendations
        if (r.get("strategy_match") or r.get("pipeline") or "unknown") in pipeline_filter
        and r.get("confidence", 0) >= min_confidence
        and r.get("final_score", 0) >= min_score
    ]

    # ---- Summary bar ----
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.5rem 0;margin-bottom:0.75rem;border-bottom:1px solid {COLORS['border']};">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                color:{COLORS['text_secondary']};">
                Showing <span style="color:{COLORS['text_primary']};font-weight:700;">
                {len(filtered)}</span> of {len(recommendations)} signals
            </span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                color:{COLORS['text_muted']};">
                {display_date}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Signal cards in 2-column grid ----
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(filtered):
                break
            rec = filtered[idx]
            ticker = rec.get("ticker", "UNKNOWN")
            rank = rec.get("rank", idx + 1)
            score = rec.get("final_score", 0)
            confidence = rec.get("confidence", 0)
            strategy = (rec.get("strategy_match") or rec.get("pipeline") or "unknown").replace("_", " ").title()
            entry_price = rec.get("entry_price", 0)
            reason = rec.get("reason", "No thesis provided.")
            company_name = rec.get("company_name", "")
            description = rec.get("description", "")
            risk_level = rec.get("risk_level", "")

            with col:
                st.markdown(
                    signal_card(
                        rank,
                        ticker,
                        score,
                        confidence,
                        strategy,
                        entry_price,
                        reason,
                        company_name,
                        description,
                        risk_level,
                    ),
                    unsafe_allow_html=True,
                )

                if show_charts:
                    st.markdown(
                        f"""
                        <div style="margin-top:0.35rem;margin-bottom:0.25rem;
                            font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                            text-transform:uppercase;letter-spacing:0.06em;
                            color:{COLORS['text_muted']};">
                            Chart Timeframe
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    chart_timeframe = st.radio(
                        f"Timeframe for {ticker}",
                        list(TIMEFRAME_LOOKBACK_DAYS.keys()),
                        index=3,
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"chart_tf_{ticker}_{idx}",
                    )
                    _render_single_dynamic_chart(ticker, chart_timeframe)

                # Action buttons
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("Enter Position", key=f"enter_{ticker}_{idx}"):
                        st.info(f"Position entry for {ticker} (TODO)")
                with btn_cols[1]:
                    if st.button("Watchlist", key=f"watch_{ticker}_{idx}"):
                        st.success(f"Added {ticker} to watchlist")

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
    "7D": 7,
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
}


@st.cache_data(ttl=3600)
def _load_price_history(ticker: str, period: str) -> pd.DataFrame:
    try:
        from tradingagents.dataflows.y_finance import download_history
    except Exception:
        return pd.DataFrame()

    data = download_history(
        ticker,
        period=period,
        interval="1d",
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
    date_col = "Date" if "Date" in data.columns else data.columns[0]
    close_col = "Close" if "Close" in data.columns else "Adj Close"
    if close_col not in data.columns:
        return pd.DataFrame()

    history = data[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
    history["date"] = pd.to_datetime(history["date"])
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


def _build_mini_chart(window: pd.DataFrame, timeframe: str) -> go.Figure:
    template = get_plotly_template()
    first_close = float(window["close"].iloc[0])
    last_close = float(window["close"].iloc[-1])
    line_color = COLORS["green"] if last_close >= first_close else COLORS["red"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=window["date"],
            y=window["close"],
            mode="lines",
            line=dict(color=line_color, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.08)" if line_color == COLORS["green"] else "rgba(239,68,68,0.08)",
            hovertemplate=f"{timeframe}<br>%{{x|%b %d, %Y}}<br>$%{{y:.2f}}<extra></extra>",
            name=timeframe,
        )
    )
    fig.update_layout(
        **template,
        height=140,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(42,53,72,0.28)", tickprefix="$", nticks=4)
    return fig


def _render_multi_timeframe_charts(ticker: str, selected_timeframes: list[str]) -> None:
    if not selected_timeframes:
        return

    base_history = _load_price_history(ticker, "1y")
    if base_history.empty:
        return

    ordered_timeframes = [tf for tf in TIMEFRAME_LOOKBACK_DAYS.keys() if tf in selected_timeframes]
    if not ordered_timeframes:
        return

    st.markdown(
        f"""
        <div style="margin-top:0.4rem;margin-bottom:0.45rem;padding:0.4rem 0.55rem;
            border:1px solid {COLORS['border']};border-radius:8px;
            background:linear-gradient(120deg, rgba(6,182,212,0.10), rgba(59,130,246,0.05));">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                text-transform:uppercase;letter-spacing:0.07em;color:{COLORS['text_secondary']};">
                Multi-Timeframe Price Map
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i in range(0, len(ordered_timeframes), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(ordered_timeframes):
                continue
            timeframe = ordered_timeframes[idx]
            window = _slice_history_window(base_history, timeframe)
            if window.empty:
                continue
            first_close = float(window["close"].iloc[0])
            last_close = float(window["close"].iloc[-1])
            move_text = _format_move_pct(window)
            move_color = COLORS["green"] if last_close >= first_close else COLORS["red"]
            fig = _build_mini_chart(window, timeframe)

            with col:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                        margin:0.1rem 0 0.2rem 0;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.70rem;
                            color:{COLORS['text_secondary']};letter-spacing:0.05em;">{timeframe}</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.70rem;
                            font-weight:600;color:{move_color};">{move_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig, width="stretch")


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
                (r.get("pipeline") or r.get("strategy_match") or "unknown") for r in recommendations
            )
        )
        pipeline_filter = st.multiselect("Strategy", pipelines, default=pipelines)
    with ctrl_cols[1]:
        min_confidence = st.slider("Min Confidence", 1, 10, 1)
    with ctrl_cols[2]:
        min_score = st.slider("Min Score", 0, 100, 0)
    with ctrl_cols[3]:
        show_charts = st.checkbox("Price Charts", value=False)
        if show_charts:
            selected_timeframes = st.multiselect(
                "Timeframes",
                list(TIMEFRAME_LOOKBACK_DAYS.keys()),
                default=["1M", "3M", "6M", "1Y"],
            )
        else:
            selected_timeframes = []

    # Apply filters
    filtered = [
        r
        for r in recommendations
        if (r.get("pipeline") or r.get("strategy_match") or "unknown") in pipeline_filter
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
            strategy = (rec.get("pipeline") or rec.get("strategy_match") or "unknown").title()
            entry_price = rec.get("entry_price", 0)
            reason = rec.get("reason", "No thesis provided.")

            with col:
                st.markdown(
                    signal_card(rank, ticker, score, confidence, strategy, entry_price, reason),
                    unsafe_allow_html=True,
                )

                if show_charts:
                    _render_multi_timeframe_charts(ticker, selected_timeframes)

                # Action buttons
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("Enter Position", key=f"enter_{ticker}_{idx}"):
                        st.info(f"Position entry for {ticker} (TODO)")
                with btn_cols[1]:
                    if st.button("Watchlist", key=f"watch_{ticker}_{idx}"):
                        st.success(f"Added {ticker} to watchlist")

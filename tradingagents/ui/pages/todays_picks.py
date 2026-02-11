"""
Signals page — today's recommendation cards with rich visual indicators.

Each signal is displayed as a data-dense card with strategy badges,
confidence bars, and expandable thesis sections.
"""

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from tradingagents.ui.theme import COLORS, get_plotly_template, page_header, signal_card
from tradingagents.ui.utils import load_recommendations


@st.cache_data(ttl=3600)
def _load_price_history(ticker: str, period: str) -> pd.DataFrame:
    try:
        from tradingagents.dataflows.y_finance import download_history
    except Exception:
        return pd.DataFrame()

    data = download_history(
        ticker, period=period, interval="1d", auto_adjust=True, progress=False,
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

    return data[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})


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
        pipelines = sorted(set(
            (r.get("pipeline") or r.get("strategy_match") or "unknown") for r in recommendations
        ))
        pipeline_filter = st.multiselect("Strategy", pipelines, default=pipelines)
    with ctrl_cols[1]:
        min_confidence = st.slider("Min Confidence", 1, 10, 1)
    with ctrl_cols[2]:
        min_score = st.slider("Min Score", 0, 100, 0)
    with ctrl_cols[3]:
        show_charts = st.checkbox("Price Charts", value=False)
        if show_charts:
            chart_window = st.selectbox("Window", ["1mo", "3mo", "6mo", "1y"], index=1)
        else:
            chart_window = "3mo"

    # Apply filters
    filtered = [
        r for r in recommendations
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
                    history = _load_price_history(ticker, chart_window)
                    if not history.empty:
                        template = get_plotly_template()
                        fig = px.line(history, x="date", y="close", labels={"date": "", "close": "Price"})

                        # Color line green if trending up, red if down
                        first_close = history["close"].iloc[0]
                        last_close = history["close"].iloc[-1]
                        line_color = COLORS["green"] if last_close >= first_close else COLORS["red"]

                        fig.update_traces(line=dict(color=line_color, width=1.5))
                        fig.update_layout(
                            **template,
                            height=160,
                            showlegend=False,
                        )
                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                        fig.update_xaxes(showticklabels=False, showgrid=False)
                        fig.update_yaxes(showgrid=True, gridcolor="rgba(42,53,72,0.3)", tickprefix="$")
                        st.plotly_chart(fig, width="stretch")

                # Action buttons
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("Enter Position", key=f"enter_{ticker}_{idx}"):
                        st.info(f"Position entry for {ticker} (TODO)")
                with btn_cols[1]:
                    if st.button("Watchlist", key=f"watch_{ticker}_{idx}"):
                        st.success(f"Added {ticker} to watchlist")

"""Today's recommendations."""

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from tradingagents.ui.utils import load_recommendations


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

    return data[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})


def render():
    st.title("📋 Today's Recommendations")

    today = datetime.now().strftime("%Y-%m-%d")
    recommendations, meta = load_recommendations(today, return_meta=True)

    if not recommendations:
        st.warning(f"No recommendations for {today}")
        return

    if meta.get("is_fallback") and meta.get("date"):
        st.info(f"No recommendations for {today}. Showing latest from {meta['date']}.")

    show_charts = st.checkbox("Show price charts", value=True)
    chart_window = st.selectbox(
        "Price history window",
        ["1mo", "3mo", "6mo", "1y"],
        index=1,
    )

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        pipelines = list(
            set(
                (r.get("pipeline") or r.get("strategy_match") or "unknown") for r in recommendations
            )
        )
        pipeline_filter = st.multiselect("Pipeline", pipelines, default=pipelines)
    with col2:
        min_confidence = st.slider("Min Confidence", 1, 10, 7)
    with col3:
        min_score = st.slider("Min Score", 0, 100, 70)

    # Apply filters
    filtered = [
        r
        for r in recommendations
        if (r.get("pipeline") or r.get("strategy_match") or "unknown") in pipeline_filter
        and r.get("confidence", 0) >= min_confidence
        and r.get("final_score", 0) >= min_score
    ]

    st.write(f"**{len(filtered)}** of **{len(recommendations)}** recommendations")

    # Display recommendations
    for i, rec in enumerate(filtered, 1):
        ticker = rec.get("ticker", "UNKNOWN")
        score = rec.get("final_score", 0)
        confidence = rec.get("confidence", 0)
        pipeline = (rec.get("pipeline") or rec.get("strategy_match") or "unknown").title()
        scanner = rec.get("scanner") or rec.get("strategy_match") or "unknown"
        entry_price = rec.get("entry_price")
        current_price = rec.get("current_price")

        with st.expander(
            f"#{i} {ticker} - {rec.get('company_name', '')} (Score: {score}, Conf: {confidence}/10)"
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Pipeline:** {pipeline}")
                st.write(f"**Scanner/Strategy:** {scanner}")
                if entry_price is not None:
                    st.write(f"**Entry Price:** ${entry_price:.2f}")
                if current_price is not None:
                    st.write(f"**Current Price:** ${current_price:.2f}")
                st.write(f"**Thesis:** {rec.get('reason', 'N/A')}")
                if show_charts:
                    history = _load_price_history(ticker, chart_window)
                    if history.empty:
                        st.caption("Price history unavailable.")
                    else:
                        last_close = history["close"].iloc[-1]
                        st.caption(f"Last close: ${last_close:.2f}")
                        fig = px.line(
                            history,
                            x="date",
                            y="close",
                            title=None,
                            labels={"date": "", "close": "Price"},
                        )
                        fig.update_traces(line=dict(color="#1f77b4", width=2))
                        fig.update_layout(
                            height=260,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
                            hovermode="x unified",
                        )
                        fig.update_yaxes(tickprefix="$")
                        st.plotly_chart(fig, use_container_width=True)

            with col2:
                if st.button("✅ Enter Position", key=f"enter_{ticker}"):
                    st.info("Position entry modal (TODO)")
                if st.button("👀 Watch", key=f"watch_{ticker}"):
                    st.success(f"Added {ticker} to watchlist")

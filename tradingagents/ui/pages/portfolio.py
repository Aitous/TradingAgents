"""
Portfolio page — position tracker with P/L visualization.

Shows portfolio summary KPIs and individual position rows
with color-coded P/L indicators.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from tradingagents.ui.theme import COLORS, kpi_card, page_header, pnl_color
from tradingagents.ui.utils import load_open_positions


def render():
    st.markdown(page_header("Portfolio", "Open positions & P/L tracker"), unsafe_allow_html=True)

    # ---- Add position form ----
    with st.expander("Add Position"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ticker = st.text_input("Ticker", placeholder="AAPL")
        with col2:
            entry_price = st.number_input("Entry Price", min_value=0.0, format="%.2f")
        with col3:
            shares = st.number_input("Shares", min_value=0, step=1)
        with col4:
            st.write("")
            if st.button("Add Position"):
                if ticker and entry_price > 0 and shares > 0:
                    from tradingagents.dataflows.discovery.performance.position_tracker import (
                        PositionTracker,
                    )

                    tracker = PositionTracker()
                    pos = tracker.create_position(
                        {
                            "ticker": ticker.upper(),
                            "entry_price": entry_price,
                            "shares": shares,
                            "recommendation_date": datetime.now().isoformat(),
                            "pipeline": "manual",
                            "scanner": "manual",
                            "strategy_match": "manual",
                            "confidence": 5,
                        }
                    )
                    tracker.save_position(pos)
                    st.success(f"Added {ticker.upper()}")
                    st.rerun()

    positions = load_open_positions()

    if not positions:
        st.markdown(
            f"""
            <div style="text-align:center;padding:3rem;color:{COLORS['text_muted']};
                font-family:'DM Sans',sans-serif;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">No open positions</div>
                <div style="font-size:0.85rem;">
                    Enter positions manually above or run the discovery pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ---- Portfolio summary ----
    total_invested = sum(p["entry_price"] * p.get("shares", 0) for p in positions)
    total_current = sum(p["metrics"]["current_price"] * p.get("shares", 0) for p in positions)
    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    cols = st.columns(4)
    summary_kpis = [
        ("Invested", f"${total_invested:,.0f}", "", "blue"),
        ("Current Value", f"${total_current:,.0f}", "", "blue"),
        (
            "P/L",
            f"${total_pnl:+,.0f}",
            f"{total_pnl_pct:+.1f}%",
            "green" if total_pnl >= 0 else "red",
        ),
        ("Positions", str(len(positions)), "", "amber"),
    ]
    for col, (label, value, delta, color) in zip(cols, summary_kpis):
        with col:
            st.markdown(kpi_card(label, value, delta, color), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ---- Position cards ----
    st.markdown(
        '<div class="section-title">Open Positions <span class="accent">// live</span></div>',
        unsafe_allow_html=True,
    )

    for p in positions:
        ticker = p["ticker"]
        entry = p["entry_price"]
        current = p["metrics"]["current_price"]
        shares = p.get("shares", 0)
        pnl_dollar = (current - entry) * shares
        pnl_pct = p["metrics"]["current_return"]
        days = p["metrics"]["days_held"]
        color = pnl_color(pnl_pct)

        st.markdown(
            f"""
            <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
                border-left:3px solid {color};border-radius:8px;
                padding:0.85rem 1.1rem;margin-bottom:0.5rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="display:flex;align-items:center;gap:1rem;">
                        <span style="font-family:'JetBrains Mono',monospace;
                            font-weight:700;font-size:1.05rem;
                            color:{COLORS['text_primary']};">{ticker}</span>
                        <span style="font-family:'JetBrains Mono',monospace;
                            font-size:0.72rem;color:{COLORS['text_muted']};">
                            {shares} shares &middot; {days}d
                        </span>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-family:'JetBrains Mono',monospace;
                            font-size:1rem;font-weight:700;color:{color};">
                            {pnl_pct:+.1f}%
                        </span>
                        <span style="font-family:'JetBrains Mono',monospace;
                            font-size:0.75rem;color:{COLORS['text_muted']};margin-left:0.5rem;">
                            ${pnl_dollar:+,.0f}
                        </span>
                    </div>
                </div>
                <div style="display:flex;gap:2rem;margin-top:0.4rem;">
                    <span style="font-family:'JetBrains Mono',monospace;
                        font-size:0.72rem;color:{COLORS['text_muted']};">
                        Entry <span style="color:{COLORS['text_secondary']};">${entry:.2f}</span>
                    </span>
                    <span style="font-family:'JetBrains Mono',monospace;
                        font-size:0.72rem;color:{COLORS['text_muted']};">
                        Current <span style="color:{COLORS['text_secondary']};">${current:.2f}</span>
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Data table fallback ----
    with st.expander("Detailed Table"):
        data = []
        for p in positions:
            pnl = (p["metrics"]["current_price"] - p["entry_price"]) * p.get("shares", 0)
            data.append(
                {
                    "Ticker": p["ticker"],
                    "Entry": p["entry_price"],
                    "Current": p["metrics"]["current_price"],
                    "Shares": p.get("shares", 0),
                    "P/L": pnl,
                    "P/L %": p["metrics"]["current_return"],
                    "Days": p["metrics"]["days_held"],
                }
            )
        st.dataframe(
            pd.DataFrame(data),
            width="stretch",
            hide_index=True,
            column_config={
                "Entry": st.column_config.NumberColumn(format="$%.2f"),
                "Current": st.column_config.NumberColumn(format="$%.2f"),
                "Shares": st.column_config.NumberColumn(format="%d"),
                "P/L": st.column_config.NumberColumn(format="$%+.2f"),
                "P/L %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Days": st.column_config.NumberColumn(format="%d"),
            },
        )

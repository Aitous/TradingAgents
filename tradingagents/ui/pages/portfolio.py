"""Portfolio tracker."""

from datetime import datetime

import pandas as pd
import streamlit as st

from tradingagents.ui.utils import load_open_positions


def render():
    st.title("💼 Portfolio Tracker")

    # Manual add form
    with st.expander("➕ Add Position"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ticker = st.text_input("Ticker")
        with col2:
            entry_price = st.number_input("Entry Price", min_value=0.0)
        with col3:
            shares = st.number_input("Shares", min_value=0, step=1)
        with col4:
            st.write("")  # Spacing
            if st.button("Add"):
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

    # Load positions
    positions = load_open_positions()

    if not positions:
        st.info("No open positions")
        return

    # Summary
    total_invested = sum(p["entry_price"] * p.get("shares", 0) for p in positions)
    total_current = sum(p["metrics"]["current_price"] * p.get("shares", 0) for p in positions)
    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Invested", f"${total_invested:,.2f}")
    with col2:
        st.metric("Current", f"${total_current:,.2f}")
    with col3:
        st.metric("P/L", f"${total_pnl:,.2f}", delta=f"{total_pnl_pct:+.1f}%")
    with col4:
        st.metric("Positions", len(positions))

    # Table
    st.subheader("📊 Positions")

    data = []
    for p in positions:
        pnl = (p["metrics"]["current_price"] - p["entry_price"]) * p.get("shares", 0)
        data.append(
            {
                "Ticker": p["ticker"],
                "Entry": f"${p['entry_price']:.2f}",
                "Current": f"${p['metrics']['current_price']:.2f}",
                "Shares": p.get("shares", 0),
                "P/L": f"${pnl:.2f}",
                "P/L %": f"{p['metrics']['current_return']:+.1f}%",
                "Days": p["metrics"]["days_held"],
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

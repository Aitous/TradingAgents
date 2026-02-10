"""
Home page for the Trading Agents Dashboard.

This module displays the main dashboard with overview metrics and
pipeline performance visualization.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from tradingagents.ui.utils import load_open_positions, load_statistics, load_strategy_metrics


def render() -> None:
    """
    Render the home page with overview metrics and pipeline performance.

    Displays:
    - Dashboard title
    - Warning if no statistics available
    - 4-column metric layout (Win Rate, Open Positions, Avg Return, Best Pipeline)
    - Pipeline performance scatter plot with quadrant lines
    """
    # Page title
    st.title("🎯 Trading Discovery Dashboard")

    # Load data
    stats = load_statistics()
    positions = load_open_positions()
    strategy_metrics = load_strategy_metrics()

    # Check if statistics are available
    if not stats or not stats.get("overall_7d"):
        st.warning("No statistics data available. Run the discovery pipeline to generate data.")
        return

    if not strategy_metrics:
        st.warning("No strategy performance data available yet.")
        return

    # Extract overall metrics from 7-day period
    overall_metrics = stats.get("overall_7d", {})
    win_rate_7d = overall_metrics.get("win_rate", 0)
    avg_return_7d = overall_metrics.get("avg_return", 0)
    open_positions_count = len(positions) if positions else 0

    # Find best strategy
    best_strategy = None
    best_win_rate = 0.0
    for item in strategy_metrics:
        win_rate = item.get("Win Rate", 0) or 0
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_strategy = {"name": item.get("Strategy", "unknown"), "win_rate": win_rate}

    # Display 4-column metric layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Win Rate (7d)",
            value=f"{win_rate_7d:.1f}%",
            delta=f"{win_rate_7d - 50:.1f}%" if win_rate_7d >= 50 else None,
        )

    with col2:
        st.metric(
            label="Open Positions",
            value=open_positions_count,
        )

    with col3:
        st.metric(
            label="Avg Return (7d)",
            value=f"{avg_return_7d:.2f}%",
            delta=f"{avg_return_7d:.2f}%" if avg_return_7d > 0 else None,
        )

    with col4:
        if best_strategy:
            st.metric(
                label="Best Strategy",
                value=best_strategy["name"],
                delta=f"{best_strategy['win_rate']:.1f}% WR",
            )
        else:
            st.metric(
                label="Best Strategy",
                value="N/A",
            )

    # Strategy Performance scatter plot
    st.subheader("Strategy Performance")

    if strategy_metrics:
        df = pd.DataFrame(strategy_metrics)

        # Create scatter plot with plotly
        fig = px.scatter(
            df,
            x="Win Rate",
            y="Avg Return",
            size="Count",
            color="Strategy",
            hover_name="Strategy",
            hover_data={
                "Win Rate": ":.1f",
                "Avg Return": ":.2f",
                "Count": True,
                "Strategy": False,
            },
            title="Strategy Performance Analysis",
            labels={
                "Win Rate": "Win Rate (%)",
                "Avg Return": "Avg Return (%)",
            },
        )

        # Add quadrant lines at y=0 (breakeven) and x=50 (50% win rate)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

        # Update layout for better visibility
        fig.update_layout(
            height=400,
            showlegend=True,
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No strategy data available for visualization.")

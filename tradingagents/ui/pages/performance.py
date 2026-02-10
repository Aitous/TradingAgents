"""
Performance analytics page for the Trading Agents Dashboard.

This module displays performance metrics and visualization for different scanners,
including win rates, average returns, and trading volume analysis.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from tradingagents.ui.utils import load_strategy_metrics


def render() -> None:
    """
    Render the performance analytics page.

    Displays:
    - Page title
    - Warning if no statistics available
    - Scanner Performance heatmap with scatter plot showing:
      - Win Rate (x-axis) vs Avg Return (y-axis)
      - Bubble size = Trade count
      - Color = Win Rate (RdYlGn colorscale)
      - Quadrant lines at y=0 and x=50
    """
    # Page title
    st.title("📊 Performance Analytics")

    # Load data
    strategy_metrics = load_strategy_metrics()

    # Check if data is available
    if not strategy_metrics:
        st.warning(
            "No strategy performance data available. Run performance tracking to generate data."
        )
        return

    # Strategy Performance section
    st.subheader("Strategy Performance")

    if strategy_metrics:
        df = pd.DataFrame(strategy_metrics)

        # Create scatter plot with plotly
        fig = px.scatter(
            df,
            x="Win Rate",
            y="Avg Return",
            size="Count",
            color="Win Rate",
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
            color_continuous_scale="RdYlGn",
        )

        # Add quadrant lines at y=0 (breakeven) and x=50 (50% win rate)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)

        # Update layout for better visibility
        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No strategy data available for visualization.")

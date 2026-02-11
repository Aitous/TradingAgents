"""
Performance analytics page — strategy comparison and win/loss analysis.

Shows strategy scatter plot with themed Plotly charts, per-strategy
breakdown table, and win rate distribution.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tradingagents.ui.theme import COLORS, get_plotly_template, page_header
from tradingagents.ui.utils import load_statistics, load_strategy_metrics


def render() -> None:
    """Render the performance analytics page."""
    st.markdown(
        page_header("Performance", "Strategy analytics & win/loss breakdown"),
        unsafe_allow_html=True,
    )

    strategy_metrics = load_strategy_metrics()
    stats = load_statistics()

    if not strategy_metrics:
        st.warning(
            "No performance data available yet. Run the discovery pipeline and track outcomes."
        )
        return

    template = get_plotly_template()
    df = pd.DataFrame(strategy_metrics)

    # ---- Summary KPIs ----
    total_trades = df["Count"].sum()
    avg_wr = (df["Win Rate"] * df["Count"]).sum() / total_trades if total_trades > 0 else 0
    avg_ret = (df["Avg Return"] * df["Count"]).sum() / total_trades if total_trades > 0 else 0
    n_strategies = len(df)

    cols = st.columns(4)
    summaries = [
        ("Total Trades", str(int(total_trades))),
        ("Weighted Win Rate", f"{avg_wr:.1f}%"),
        ("Weighted Avg Return", f"{avg_ret:+.2f}%"),
        ("Active Strategies", str(n_strategies)),
    ]
    for col, (label, val) in zip(cols, summaries):
        with col:
            st.markdown(
                f"""
                <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
                    border-radius:8px;padding:0.85rem 1rem;text-align:center;">
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.65rem;
                        font-weight:600;text-transform:uppercase;letter-spacing:0.06em;
                        color:{COLORS['text_muted']};margin-bottom:0.3rem;">{label}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                        font-weight:700;color:{COLORS['text_primary']};">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ---- Two-column: scatter + bar chart ----
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(
            '<div class="section-title">Win Rate vs Return <span class="accent">// scatter</span></div>',
            unsafe_allow_html=True,
        )

        fig = px.scatter(
            df,
            x="Win Rate",
            y="Avg Return",
            size="Count",
            color="Win Rate",
            hover_name="Strategy",
            hover_data={"Win Rate": ":.1f", "Avg Return": ":.2f", "Count": True},
            labels={"Win Rate": "Win Rate (%)", "Avg Return": "Avg Return (%)"},
            color_continuous_scale=[
                [0, COLORS["red"]],
                [0.5, COLORS["amber"]],
                [1.0, COLORS["green"]],
            ],
            size_max=45,
        )

        fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], opacity=0.4)
        fig.add_vline(x=50, line_dash="dot", line_color=COLORS["text_muted"], opacity=0.4)

        fig.update_layout(
            **template,
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, width="stretch")

    with right_col:
        st.markdown(
            '<div class="section-title">Win Rate by Strategy <span class="accent">// bar</span></div>',
            unsafe_allow_html=True,
        )

        df_sorted = df.sort_values("Win Rate", ascending=True)
        colors = [COLORS["green"] if wr >= 50 else COLORS["red"] for wr in df_sorted["Win Rate"]]

        fig_bar = go.Figure(
            go.Bar(
                x=df_sorted["Win Rate"],
                y=df_sorted["Strategy"],
                orientation="h",
                marker_color=colors,
                text=[f"{wr:.0f}%" for wr in df_sorted["Win Rate"]],
                textposition="auto",
                textfont=dict(family="JetBrains Mono", size=11, color=COLORS["text_primary"]),
            )
        )

        fig_bar.add_vline(x=50, line_dash="dot", line_color=COLORS["text_muted"], opacity=0.5)

        fig_bar.update_layout(
            **template,
            height=400,
            xaxis_title="Win Rate (%)",
            yaxis_title="",
        )
        fig_bar.update_yaxes(tickfont=dict(family="JetBrains Mono", size=11))
        st.plotly_chart(fig_bar, width="stretch")

    # ---- Strategy breakdown table ----
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Detailed Breakdown <span class="accent">// table</span></div>',
        unsafe_allow_html=True,
    )

    display_df = df.copy()
    display_df = display_df.sort_values("Win Rate", ascending=False)
    display_df["Count"] = display_df["Count"].astype(int)
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Avg Return": st.column_config.NumberColumn(format="%+.2f%%"),
            "Count": st.column_config.NumberColumn(format="%d"),
        },
    )

    # ---- Per-strategy stats from statistics.json ----
    if stats and stats.get("by_strategy"):
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Time-Period Breakdown <span class="accent">// 1d / 7d / 30d</span></div>',
            unsafe_allow_html=True,
        )

        by_strat = stats["by_strategy"]
        rows = []
        for strat_name, data in by_strat.items():
            rows.append(
                {
                    "Strategy": strat_name,
                    "Count": data.get("count", 0),
                    "Win Rate 1d": (
                        f"{data.get('win_rate_1d', 0):.0f}%" if "win_rate_1d" in data else "N/A"
                    ),
                    "Win Rate 7d": (
                        f"{data.get('win_rate_7d', 0):.0f}%" if "win_rate_7d" in data else "N/A"
                    ),
                    "Wins 1d": data.get("wins_1d", 0),
                    "Losses 1d": data.get("losses_1d", 0),
                    "Wins 7d": data.get("wins_7d", 0),
                    "Losses 7d": data.get("losses_7d", 0),
                }
            )

        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

"""
Performance analytics page — strategy comparison and win/loss analysis.

Shows strategy scatter plot with themed Plotly charts, per-strategy
breakdown table, win rate distribution, and full recommendation history.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tradingagents.ui.theme import COLORS, get_plotly_template, page_header, pnl_color
from tradingagents.ui.utils import (
    load_performance_database,
    load_statistics,
    load_strategy_metrics,
)


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
    # Weighted averages only over strategies that have evaluated data (non-NaN)
    eval_df = df.dropna(subset=["Win Rate", "Avg Return"])
    eval_trades = eval_df["Count"].sum()
    avg_wr = (eval_df["Win Rate"] * eval_df["Count"]).sum() / eval_trades if eval_trades > 0 else 0
    avg_ret = (
        (eval_df["Avg Return"] * eval_df["Count"]).sum() / eval_trades if eval_trades > 0 else 0
    )
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

        df_bar = df.dropna(subset=["Win Rate"]).sort_values("Win Rate", ascending=True)
        colors = [COLORS["green"] if wr >= 50 else COLORS["red"] for wr in df_bar["Win Rate"]]

        fig_bar = go.Figure(
            go.Bar(
                x=df_bar["Win Rate"],
                y=df_bar["Strategy"],
                orientation="h",
                marker_color=colors,
                text=[f"{wr:.0f}%" for wr in df_bar["Win Rate"]],
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
                    "Win Rate 1d": data.get("win_rate_1d") if "win_rate_1d" in data else None,
                    "Avg Ret 1d": data.get("avg_return_1d") if "avg_return_1d" in data else None,
                    "W/L 1d": (
                        f"{data.get('wins_1d', 0)}W/{data.get('losses_1d', 0)}L"
                        if data.get("wins_1d", 0) + data.get("losses_1d", 0) > 0
                        else "—"
                    ),
                    "Win Rate 7d": data.get("win_rate_7d") if "win_rate_7d" in data else None,
                    "Avg Ret 7d": data.get("avg_return_7d") if "avg_return_7d" in data else None,
                    "W/L 7d": (
                        f"{data.get('wins_7d', 0)}W/{data.get('losses_7d', 0)}L"
                        if data.get("wins_7d", 0) + data.get("losses_7d", 0) > 0
                        else "—"
                    ),
                }
            )

        if rows:
            period_df = pd.DataFrame(rows).sort_values("Count", ascending=False)
            st.dataframe(
                period_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Count": st.column_config.NumberColumn(format="%d"),
                    "Win Rate 1d": st.column_config.NumberColumn(format="%.1f%%"),
                    "Avg Ret 1d": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Win Rate 7d": st.column_config.NumberColumn(format="%.1f%%"),
                    "Avg Ret 7d": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )

    # ---- Recommendation History ----
    _render_recommendation_history(template)


# ---------------------------------------------------------------------------
# Recommendation history helpers
# ---------------------------------------------------------------------------


def _return_cell(val) -> str:
    """Format a return value as a colored HTML span."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<span style="color:{c};">—</span>'.format(c=COLORS["text_muted"])
    color = pnl_color(val)
    return f'<span style="color:{color};font-weight:600;">{val:+.2f}%</span>'


def _win_dot(val) -> str:
    """Green/red dot for win/loss boolean."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    color = COLORS["green"] if val else COLORS["red"]
    return f'<span style="color:{color};font-size:0.7rem;">●</span>'


def _render_recommendation_history(template: dict) -> None:
    """Full recommendation history with charts and filterable table."""
    recs = load_performance_database()
    if not recs:
        return

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Recommendation History '
        '<span class="accent">// all picks</span></div>',
        unsafe_allow_html=True,
    )

    # Build DataFrame
    hist_df = pd.DataFrame(recs)

    # Ensure numeric types
    for col in ["return_1d", "return_7d", "return_30d", "return_pct", "final_score", "confidence"]:
        if col in hist_df.columns:
            hist_df[col] = pd.to_numeric(hist_df[col], errors="coerce")

    # Parse dates
    if "discovery_date" in hist_df.columns:
        hist_df["discovery_date"] = pd.to_datetime(hist_df["discovery_date"], errors="coerce")

    # ---- Filters row ----
    filter_cols = st.columns([2, 2, 2, 1])

    with filter_cols[0]:
        strategies = sorted(hist_df["strategy_match"].dropna().unique())
        selected_strategies = st.multiselect(
            "Strategy",
            strategies,
            default=[],
            placeholder="All strategies",
        )

    with filter_cols[1]:
        dates = hist_df["discovery_date"].dropna().sort_values()
        min_date = dates.min().date() if len(dates) > 0 else None
        max_date = dates.max().date() if len(dates) > 0 else None
        if min_date and max_date:
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
        else:
            date_range = None

    with filter_cols[2]:
        outcome_filter = st.selectbox(
            "Outcome (7d)",
            ["All", "Winners", "Losers", "Pending"],
            index=0,
        )

    with filter_cols[3]:
        sort_by = st.selectbox("Sort", ["Date", "Return 1d", "Return 7d", "Score"], index=0)

    # Apply filters
    mask = pd.Series(True, index=hist_df.index)

    if selected_strategies:
        mask &= hist_df["strategy_match"].isin(selected_strategies)

    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask &= (hist_df["discovery_date"] >= start) & (hist_df["discovery_date"] <= end)

    if outcome_filter == "Winners":
        mask &= hist_df.get("win_7d", pd.Series(dtype=bool)) == True  # noqa: E712
    elif outcome_filter == "Losers":
        mask &= hist_df.get("win_7d", pd.Series(dtype=bool)) == False  # noqa: E712
    elif outcome_filter == "Pending":
        mask &= hist_df.get("return_7d").isna() if "return_7d" in hist_df.columns else True

    filtered = hist_df[mask].copy()

    # Sort
    sort_map = {
        "Date": ("discovery_date", False),
        "Return 1d": ("return_1d", False),
        "Return 7d": ("return_7d", False),
        "Score": ("final_score", False),
    }
    sort_col, sort_asc = sort_map.get(sort_by, ("discovery_date", False))
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=sort_asc, na_position="last")

    st.caption(f"Showing {len(filtered)} of {len(hist_df)} recommendations")

    # ---- Two-column charts ----
    if len(filtered) > 0:
        left_ch, right_ch = st.columns(2)

        with left_ch:
            st.markdown(
                '<div class="section-title">Return Distribution '
                '<span class="accent">// 1d vs 7d</span></div>',
                unsafe_allow_html=True,
            )
            _render_return_distribution(filtered, template)

        with right_ch:
            st.markdown(
                '<div class="section-title">Cumulative P/L by Date '
                '<span class="accent">// equity curve</span></div>',
                unsafe_allow_html=True,
            )
            _render_cumulative_pnl(filtered, template)

    # ---- Full history table ----
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">All Picks ' '<span class="accent">// detail table</span></div>',
        unsafe_allow_html=True,
    )
    _render_history_table(filtered)


def _render_return_distribution(df: pd.DataFrame, template: dict) -> None:
    """Box plot comparing 1d vs 7d return distributions."""
    ret_data = []
    for _, row in df.iterrows():
        if pd.notna(row.get("return_1d")):
            ret_data.append({"Period": "1-Day", "Return (%)": row["return_1d"]})
        if pd.notna(row.get("return_7d")):
            ret_data.append({"Period": "7-Day", "Return (%)": row["return_7d"]})

    if not ret_data:
        st.info("No return data available for the selected filters.")
        return

    ret_df = pd.DataFrame(ret_data)

    fig = go.Figure()
    for period, color in [("1-Day", COLORS["blue"]), ("7-Day", COLORS["cyan"])]:
        subset = ret_df[ret_df["Period"] == period]["Return (%)"]
        if len(subset) == 0:
            continue
        fig.add_trace(
            go.Box(
                y=subset,
                name=period,
                marker_color=color,
                boxmean=True,
                jitter=0.3,
                pointpos=-1.5,
                boxpoints="outliers",
            )
        )

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], opacity=0.4)
    fig.update_layout(
        **template,
        height=350,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        yaxis_title="Return (%)",
    )
    st.plotly_chart(fig, width="stretch")


def _render_cumulative_pnl(df: pd.DataFrame, template: dict) -> None:
    """Cumulative average return by discovery date (equity curve style)."""
    if "discovery_date" not in df.columns:
        st.info("No date data available.")
        return

    # Use 7d return where available, fall back to 1d
    df_dated = df.dropna(subset=["discovery_date"]).copy()
    df_dated["best_return"] = df_dated["return_7d"].fillna(df_dated.get("return_1d", 0))
    df_dated = df_dated.dropna(subset=["best_return"])

    if len(df_dated) == 0:
        st.info("No return data available for equity curve.")
        return

    # Group by date, get mean return per day
    daily = (
        df_dated.groupby("discovery_date")["best_return"]
        .mean()
        .reset_index()
        .sort_values("discovery_date")
    )
    daily.columns = ["Date", "Avg Return"]
    daily["Cumulative"] = daily["Avg Return"].cumsum()

    # Color based on cumulative being positive/negative
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in daily["Cumulative"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Cumulative"],
            mode="lines+markers",
            line=dict(color=COLORS["green"], width=2),
            marker=dict(color=colors, size=7, line=dict(color=COLORS["bg_card"], width=1)),
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.08)",
            hovertemplate="Date: %{x|%b %d}<br>Cumulative: %{y:+.2f}%<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], opacity=0.4)
    fig.update_layout(
        **template,
        height=350,
        showlegend=False,
        yaxis_title="Cumulative Avg Return (%)",
        xaxis_title="",
    )
    st.plotly_chart(fig, width="stretch")


def _render_history_table(df: pd.DataFrame) -> None:
    """Render the full recommendation history as a styled dataframe."""
    if len(df) == 0:
        st.info("No recommendations match the selected filters.")
        return

    # Build display dataframe with readable columns
    display_rows = []
    for _, row in df.iterrows():
        disc_date = row.get("discovery_date")
        date_str = disc_date.strftime("%Y-%m-%d") if pd.notna(disc_date) else "—"

        display_rows.append(
            {
                "Date": date_str,
                "Ticker": row.get("ticker", "—"),
                "#": int(row["rank"]) if pd.notna(row.get("rank")) else 0,
                "Strategy": row.get("strategy_match", "—"),
                "Score": row.get("final_score"),
                "Conf": int(row["confidence"]) if pd.notna(row.get("confidence")) else None,
                "Entry $": row.get("entry_price"),
                "Now $": row.get("current_price"),
                "Ret 1d %": row.get("return_1d"),
                "Ret 7d %": row.get("return_7d"),
                "Ret 30d %": row.get("return_30d") if "return_30d" in row.index else None,
                "Current %": row.get("return_pct"),
                "Days": int(row["days_held"]) if pd.notna(row.get("days_held")) else None,
                "Status": row.get("status", "—"),
            }
        )

    table_df = pd.DataFrame(display_rows)

    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        height=min(len(table_df) * 35 + 38, 600),
        column_config={
            "Date": st.column_config.TextColumn(width="small"),
            "Ticker": st.column_config.TextColumn(width="small"),
            "#": st.column_config.NumberColumn(format="%d", width="small"),
            "Strategy": st.column_config.TextColumn(width="medium"),
            "Score": st.column_config.NumberColumn(format="%.0f", width="small"),
            "Conf": st.column_config.NumberColumn(format="%d/10", width="small"),
            "Entry $": st.column_config.NumberColumn(format="$%.2f"),
            "Now $": st.column_config.NumberColumn(format="$%.2f"),
            "Ret 1d %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Ret 7d %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Ret 30d %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Current %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Days": st.column_config.NumberColumn(format="%d"),
            "Status": st.column_config.TextColumn(width="small"),
        },
    )

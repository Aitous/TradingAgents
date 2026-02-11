"""
Overview page — trading terminal home screen.

Shows KPI cards, strategy scatter plot, and recent signal summary.
"""

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from tradingagents.ui.theme import COLORS, get_plotly_template, kpi_card, page_header
from tradingagents.ui.utils import (
    load_open_positions,
    load_recommendations,
    load_statistics,
    load_strategy_metrics,
)


def render() -> None:
    """Render the overview page."""

    st.markdown(
        page_header("Overview", f"Market session {datetime.now().strftime('%A, %B %d %Y')}"),
        unsafe_allow_html=True,
    )

    stats = load_statistics()
    positions = load_open_positions()
    strategy_metrics = load_strategy_metrics()

    overall = stats.get("overall_7d", {}) if stats else {}
    win_rate_7d = overall.get("win_rate", 0)
    avg_return_7d = overall.get("avg_return", 0)
    total_recs = stats.get("total_recommendations", 0) if stats else 0
    open_count = len(positions) if positions else 0

    best_strat_name = "N/A"
    best_strat_wr = 0.0
    for item in (strategy_metrics or []):
        wr = item.get("Win Rate", 0) or 0
        if wr > best_strat_wr:
            best_strat_wr = wr
            best_strat_name = item.get("Strategy", "unknown")

    # ---- KPI Row ----
    cols = st.columns(5)
    kpis = [
        ("Win Rate 7d", f"{win_rate_7d:.0f}%", f"+{win_rate_7d - 50:.0f}pp vs 50%" if win_rate_7d >= 50 else f"{win_rate_7d - 50:.0f}pp vs 50%", "green" if win_rate_7d >= 50 else "red"),
        ("Avg Return 7d", f"{avg_return_7d:+.2f}%", "", "green" if avg_return_7d > 0 else "red"),
        ("Open Positions", str(open_count), "", "blue"),
        ("Total Signals", str(total_recs), "", "amber"),
        ("Top Strategy", best_strat_name.upper(), f"{best_strat_wr:.0f}% WR" if best_strat_wr else "", "green" if best_strat_wr >= 60 else "amber"),
    ]
    for col, (label, value, delta, color) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(label, value, delta, color), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ---- Two-column: strategy chart + today's signals ----
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown(
            '<div class="section-title">Strategy Performance <span class="accent">// scatter</span></div>',
            unsafe_allow_html=True,
        )

        if strategy_metrics:
            df = pd.DataFrame(strategy_metrics)
            template = get_plotly_template()

            fig = px.scatter(
                df,
                x="Win Rate",
                y="Avg Return",
                size="Count",
                color="Strategy",
                hover_name="Strategy",
                hover_data={"Win Rate": ":.1f", "Avg Return": ":.2f", "Count": True, "Strategy": False},
                labels={"Win Rate": "Win Rate (%)", "Avg Return": "Avg Return (%)"},
                size_max=40,
            )

            fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], opacity=0.4)
            fig.add_vline(x=50, line_dash="dot", line_color=COLORS["text_muted"], opacity=0.4)
            fig.add_annotation(x=75, y=5, text="WINNERS", showarrow=False, font=dict(size=10, color=COLORS["green"], family="JetBrains Mono"), opacity=0.3)
            fig.add_annotation(x=25, y=-5, text="LOSERS", showarrow=False, font=dict(size=10, color=COLORS["red"], family="JetBrains Mono"), opacity=0.3)

            fig.update_layout(
                **template,
                height=380,
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h", yanchor="bottom", y=-0.25),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Run the discovery pipeline to generate strategy data.")

    with right_col:
        st.markdown(
            '<div class="section-title">Today\'s Signals <span class="accent">// latest</span></div>',
            unsafe_allow_html=True,
        )

        today = datetime.now().strftime("%Y-%m-%d")
        recs = load_recommendations(today)

        if recs:
            for rec in recs[:6]:
                ticker = rec.get("ticker", "???")
                score = rec.get("final_score", 0)
                conf = rec.get("confidence", 0)
                strat = (rec.get("strategy_match") or "momentum").upper()
                entry = rec.get("entry_price")
                entry_str = f"${entry:.2f}" if entry else "N/A"

                score_color = COLORS["green"] if score >= 35 else (COLORS["amber"] if score >= 20 else COLORS["text_muted"])
                conf_bar_w = conf * 10
                conf_color = COLORS["green"] if conf >= 8 else (COLORS["amber"] if conf >= 6 else COLORS["red"])

                st.markdown(
                    f"""
                    <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
                        border-radius:8px;padding:0.65rem 0.85rem;margin-bottom:0.5rem;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div style="display:flex;align-items:center;gap:0.6rem;">
                                <span style="font-family:'JetBrains Mono',monospace;
                                    font-weight:700;font-size:0.95rem;
                                    color:{COLORS['text_primary']};">{ticker}</span>
                                <span style="font-family:'DM Sans',sans-serif;font-size:0.6rem;
                                    font-weight:600;text-transform:uppercase;
                                    padding:0.15rem 0.4rem;border-radius:3px;
                                    background:rgba(59,130,246,0.15);
                                    color:{COLORS['blue']};letter-spacing:0.04em;">{strat}</span>
                            </div>
                            <span style="font-family:'JetBrains Mono',monospace;
                                font-size:0.8rem;color:{score_color};font-weight:600;">{score}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;
                            align-items:center;margin-top:0.35rem;">
                            <span style="font-family:'JetBrains Mono',monospace;
                                font-size:0.72rem;color:{COLORS['text_muted']};">{entry_str}</span>
                            <div style="width:50px;height:3px;background:{COLORS['border']};
                                border-radius:2px;overflow:hidden;">
                                <div style="height:100%;width:{conf_bar_w}%;
                                    background:{conf_color};border-radius:2px;"></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if len(recs) > 6:
                st.caption(f"+{len(recs) - 6} more signals. Switch to Signals page for the full list.")
        else:
            st.info("No signals generated today.")

    # ---- Strategy table ----
    if strategy_metrics:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Strategy Breakdown <span class="accent">// table</span></div>',
            unsafe_allow_html=True,
        )
        df_table = pd.DataFrame(strategy_metrics).sort_values("Win Rate", ascending=False)
        st.dataframe(
            df_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                "Avg Return": st.column_config.NumberColumn(format="%+.2f%%"),
                "Count": st.column_config.NumberColumn(format="%d"),
            },
        )

"""
Overview page — Morning Briefing.

Answers three questions without requiring page navigation:
  1. What signals have conviction today?
  2. Is the system healthy (scanners performing)?
  3. How are strategies trending?
"""

import html as _html
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from tradingagents.ui.theme import COLORS, get_plotly_template, kpi_card
from tradingagents.ui.utils import (
    load_open_positions,
    load_recommendations,
    load_scanner_health,
    load_statistics,
    load_strategy_metrics,
)

_HEALTH_DOT = {
    "green": f'<span style="color:{COLORS["green"]};font-size:0.7rem;">●</span>',
    "amber": f'<span style="color:{COLORS["amber"]};font-size:0.7rem;">●</span>',
    "red": f'<span style="color:{COLORS["red"]};font-size:0.7rem;">●</span>',
    "gray": f'<span style="color:{COLORS["text_muted"]};font-size:0.7rem;">●</span>',
}


def _conviction(score: float, conf: float) -> int:
    return round(score * conf / 10)


def _conviction_color(cv: int) -> str:
    if cv >= 35:
        return COLORS["green"]
    if cv >= 22:
        return COLORS["amber"]
    return COLORS["text_muted"]


def _strategy_badge(strategy: str) -> str:
    s = strategy.lower()
    if "momentum" in s:
        bg, color = "rgba(34,197,94,0.12)", COLORS["green"]
    elif "insider" in s:
        bg, color = "rgba(245,158,11,0.12)", COLORS["amber"]
    elif "earnings" in s:
        bg, color = "rgba(59,130,246,0.12)", COLORS["blue"]
    elif "news" in s:
        bg, color = "rgba(6,182,212,0.12)", COLORS["cyan"]
    else:
        bg, color = "rgba(100,116,139,0.12)", COLORS["text_secondary"]
    label = _html.escape(strategy.upper()[:16])
    return (
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.6rem;font-weight:600;'
        f"padding:0.15rem 0.45rem;border-radius:3px;text-transform:uppercase;"
        f'letter-spacing:0.04em;background:{bg};color:{color};">{label}</span>'
    )


def _briefing_signal(rank: int, rec: dict) -> str:
    ticker = _html.escape(rec.get("ticker", "???"))
    score = rec.get("final_score", 0) or 0
    conf = rec.get("confidence", 0) or 0
    strategy = (rec.get("strategy_match") or "unknown")
    entry = rec.get("entry_price")
    entry_str = f"${entry:.2f}" if entry else "—"
    thesis = _html.escape((rec.get("reason") or rec.get("thesis") or "").strip())
    thesis_short = thesis[:110] + ("…" if len(thesis) > 110 else "")
    company = _html.escape(rec.get("company_name") or "")

    cv = _conviction(score, conf)
    cv_color = _conviction_color(cv)
    badge = _strategy_badge(strategy)
    rank_str = f"#{rank}"

    company_line = (
        f'<div style="font-size:0.7rem;color:{COLORS["text_muted"]};margin-bottom:0.35rem;">'
        f"{company}</div>"
        if company and company != ticker
        else ""
    )

    thesis_block = (
        f'<div style="font-size:0.78rem;line-height:1.45;color:{COLORS["text_secondary"]};'
        f'margin-top:0.45rem;font-family:\'DM Sans\',sans-serif;">{thesis_short}</div>'
        if thesis_short
        else ""
    )

    return (
        f'<div style="background:{COLORS["bg_card"]};border:1px solid {COLORS["border"]};'
        f"border-radius:9px;padding:0.85rem 1rem;margin-bottom:0.6rem;"
        f'transition:border-color 0.15s;">'
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:0.3rem;">'
        f'<div style="display:flex;align-items:center;gap:0.55rem;">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;'
        f'color:{COLORS["text_muted"]};font-weight:600;">{rank_str}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-weight:700;'
        f'font-size:1.05rem;color:{COLORS["text_primary"]};">{ticker}</span>'
        f"{badge}"
        f"</div>"
        f'<div style="text-align:right;">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1rem;'
        f'font-weight:700;color:{cv_color};">{cv}</div>'
        f'<div style="font-size:0.58rem;color:{COLORS["text_muted"]};text-transform:uppercase;'
        f'letter-spacing:0.06em;">conviction</div>'
        f"</div>"
        f"</div>"
        f"{company_line}"
        f'<div style="display:flex;gap:1.2rem;">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
        f'color:{COLORS["text_muted"]};">score {score}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
        f'color:{COLORS["text_muted"]};">conf {conf}/10</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
        f'color:{COLORS["text_muted"]};">{entry_str}</span>'
        f"</div>"
        f"{thesis_block}"
        f"</div>"
    )


def render() -> None:
    recs, meta = load_recommendations(return_meta=True)
    recs_date = meta.get("date") or datetime.now().strftime("%Y-%m-%d")
    is_fallback = meta.get("is_fallback", False)

    try:
        parsed = datetime.strptime(recs_date, "%Y-%m-%d")
        date_display = parsed.strftime("%A, %B %d")
    except ValueError:
        date_display = recs_date

    stats = load_statistics()
    positions = load_open_positions()
    strategy_metrics = load_strategy_metrics()
    scanner_health = load_scanner_health()

    overall = stats.get("overall_7d", {}) if stats else {}
    win_rate_7d = overall.get("win_rate", 0) or 0
    avg_return_7d = overall.get("avg_return", 0) or 0
    open_count = len(positions) if positions else 0
    n_signals = len(recs)
    n_scanners = len(scanner_health)

    best_strat_name, best_strat_wr = "N/A", 0.0
    for item in strategy_metrics or []:
        wr = item.get("Win Rate") or 0
        if wr > best_strat_wr:
            best_strat_wr = wr
            best_strat_name = item.get("Strategy", "unknown")

    # ── Briefing header ────────────────────────────────────────────────────
    fallback_pill = (
        f'<span style="background:rgba(245,158,11,0.12);color:{COLORS["amber"]};'
        f'font-size:0.65rem;font-weight:600;padding:0.1rem 0.45rem;border-radius:3px;'
        f'border:1px solid {COLORS["amber_dim"]};">showing {recs_date}</span>'
        if is_fallback
        else ""
    )
    context_parts = []
    if n_signals:
        context_parts.append(f"{n_signals} signals")
    if n_scanners:
        context_parts.append(f"{n_scanners} scanners active")
    context_str = " · ".join(context_parts) if context_parts else "no signals yet"

    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;padding-bottom:1rem;
            border-bottom:1px solid {COLORS['border']};">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:600;
                text-transform:uppercase;letter-spacing:0.12em;color:{COLORS['green']};
                margin-bottom:0.3rem;">Morning Briefing</div>
            <h1 style="font-family:'DM Sans',sans-serif;font-weight:700;font-size:1.75rem;
                color:{COLORS['text_primary']};margin:0;letter-spacing:-0.03em;
                line-height:1.15;">{date_display}</h1>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                color:{COLORS['text_muted']};margin-top:0.4rem;display:flex;
                align-items:center;gap:0.75rem;">
                <span>{context_str}</span>
                {fallback_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card(
                "Win Rate 7d",
                f"{win_rate_7d:.0f}%",
                f"+{win_rate_7d - 50:.0f}pp vs 50%" if win_rate_7d >= 50 else f"{win_rate_7d - 50:.0f}pp vs 50%",
                "green" if win_rate_7d >= 50 else "red",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card("Avg Return 7d", f"{avg_return_7d:+.2f}%", "", "green" if avg_return_7d > 0 else "red"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(kpi_card("Open Positions", str(open_count), "", "blue"), unsafe_allow_html=True)
    with c4:
        st.markdown(
            kpi_card(
                "Top Strategy",
                best_strat_name.upper()[:12],
                f"{best_strat_wr:.0f}% WR" if best_strat_wr else "",
                "green" if best_strat_wr >= 60 else "amber",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Main two-column section ────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2])

    # ---- Left: Conviction Plays ----
    with left_col:
        st.markdown(
            f'<div class="section-title">Conviction Plays'
            f'<span class="accent">// today\'s top picks</span></div>',
            unsafe_allow_html=True,
        )

        if recs:
            ranked = sorted(
                recs,
                key=lambda r: _conviction(r.get("final_score", 0) or 0, r.get("confidence", 0) or 0),
                reverse=True,
            )
            for i, rec in enumerate(ranked[:5], start=1):
                st.markdown(_briefing_signal(i, rec), unsafe_allow_html=True)

            remaining = len(recs) - 5
            if remaining > 0:
                st.caption(f"+{remaining} more — open Signals for the full list.")
        else:
            st.markdown(
                f'<div style="padding:1.5rem;background:{COLORS["bg_card"]};'
                f'border:1px solid {COLORS["border"]};border-radius:9px;'
                f'text-align:center;color:{COLORS["text_muted"]};'
                f'font-size:0.82rem;">No signals generated yet — run the discovery pipeline.</div>',
                unsafe_allow_html=True,
            )

    # ---- Right: System Health ----
    with right_col:
        # Scanner health panel
        st.markdown(
            '<div class="section-title">Scanner Health</div>',
            unsafe_allow_html=True,
        )

        if scanner_health:
            rows_html = ""
            for s in scanner_health[:8]:
                dot = _HEALTH_DOT[s["health"]]
                name = _html.escape(s["scanner"].replace("_", " ").title())
                wr = s["win_rate_1d"]
                wr_str = f"{wr:.0f}%" if wr is not None else "—"
                total = s["total"]
                rows_html += (
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'padding:0.4rem 0;border-bottom:1px solid rgba(42,53,72,0.4);">'
                    f'<div style="display:flex;align-items:center;gap:0.5rem;">'
                    f"{dot}"
                    f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.78rem;'
                    f'color:{COLORS["text_secondary"]};">{name}</span>'
                    f"</div>"
                    f'<div style="display:flex;align-items:center;gap:0.75rem;">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
                    f'color:{COLORS["text_primary"]};font-weight:600;">{wr_str}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
                    f'color:{COLORS["text_muted"]};">{total}px</span>'
                    f"</div>"
                    f"</div>"
                )
            st.markdown(
                f'<div style="background:{COLORS["bg_card"]};border:1px solid {COLORS["border"]};'
                f'border-radius:9px;padding:0.75rem 1rem;">{rows_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No scanner data yet.")

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        # Strategy performance summary
        if strategy_metrics:
            st.markdown(
                '<div class="section-title">Strategy Breakdown</div>',
                unsafe_allow_html=True,
            )
            sorted_strats = sorted(
                [s for s in strategy_metrics if s.get("Win Rate") is not None],
                key=lambda x: x["Win Rate"],
                reverse=True,
            )
            rows_html = ""
            for s in sorted_strats[:5]:
                name = _html.escape(s["Strategy"].replace("_", " ").title())
                wr = s["Win Rate"]
                ret = s.get("Avg Return") or 0
                count = s.get("Count", 0)
                wr_color = COLORS["green"] if wr >= 55 else (COLORS["amber"] if wr >= 45 else COLORS["red"])
                ret_color = COLORS["green"] if ret > 0 else (COLORS["red"] if ret < 0 else COLORS["text_muted"])
                bar_pct = min(wr, 100)
                rows_html += (
                    f'<div style="margin-bottom:0.6rem;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:baseline;margin-bottom:0.2rem;">'
                    f'<span style="font-size:0.75rem;color:{COLORS["text_secondary"]};'
                    f'font-family:\'DM Sans\',sans-serif;">{name}</span>'
                    f'<div style="display:flex;gap:0.6rem;">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
                    f'font-weight:700;color:{wr_color};">{wr:.0f}%</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
                    f'color:{ret_color};">{ret:+.1f}%</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
                    f'color:{COLORS["text_muted"]};">n={count}</span>'
                    f"</div>"
                    f"</div>"
                    f'<div style="height:3px;background:{COLORS["border"]};border-radius:2px;">'
                    f'<div style="height:100%;width:{bar_pct}%;background:{wr_color};'
                    f'border-radius:2px;"></div>'
                    f"</div>"
                    f"</div>"
                )
            st.markdown(
                f'<div style="background:{COLORS["bg_card"]};border:1px solid {COLORS["border"]};'
                f'border-radius:9px;padding:0.75rem 1rem;">{rows_html}</div>',
                unsafe_allow_html=True,
            )

    # ── Strategy scatter (secondary reference) ────────────────────────────
    if strategy_metrics:
        st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Strategy Performance'
            '<span class="accent">// score vs return</span></div>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame(strategy_metrics).dropna(subset=["Win Rate", "Avg Return"])
        if not df.empty:
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
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], opacity=0.35)
            fig.add_vline(x=50, line_dash="dot", line_color=COLORS["text_muted"], opacity=0.35)
            fig.update_layout(
                **template,
                height=300,
                showlegend=True,
                legend=dict(
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10),
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                ),
            )
            st.plotly_chart(fig, width="stretch")

"""
Scanner Performance page — per-scanner win rates, pick volume, and ranker lift.

Data sources:
  data/scanner_picks/YYYY-MM-DD.json  — one row per (ticker, scanner)
  data/discovery_events/YYYY-MM-DD.json — one row per ticker (ranker input set)

Forward returns are backfilled nightly by track_recommendation_performance.py.
Until 7–30 days have elapsed, win rate cells show N/A.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tradingagents.ui.theme import COLORS, get_plotly_template, page_header, pnl_color
from tradingagents.ui.utils import get_data_directory

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_all_json(directory: Path, key: str) -> List[Dict[str, Any]]:
    """Load all rows from date-partitioned JSON files in a directory."""
    rows = []
    if not directory.exists():
        return rows
    for f in sorted(directory.glob("*.json")):
        try:
            import json

            data = json.loads(f.read_text())
            rows.extend(data.get(key, []))
        except Exception:
            continue
    return rows


def load_scanner_picks() -> List[Dict[str, Any]]:
    return _load_all_json(get_data_directory() / "scanner_picks", "picks")


def load_discovery_events() -> List[Dict[str, Any]]:
    return _load_all_json(get_data_directory() / "discovery_events", "events")


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _scanner_metrics(picks: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-scanner win rates and pick counts."""
    agg: Dict[str, Dict] = defaultdict(
        lambda: {
            "total": 0,
            "evaluated_7d": 0,
            "wins_7d": 0,
            "sum_ret_7d": 0.0,
            "evaluated_30d": 0,
            "wins_30d": 0,
            "sum_ret_30d": 0.0,
        }
    )
    for p in picks:
        s = p.get("scanner", "unknown")
        agg[s]["total"] += 1
        if p.get("return_7d") is not None:
            agg[s]["evaluated_7d"] += 1
            agg[s]["sum_ret_7d"] += float(p["return_7d"])
            if p.get("win_7d"):
                agg[s]["wins_7d"] += 1
        if p.get("return_30d") is not None:
            agg[s]["evaluated_30d"] += 1
            agg[s]["sum_ret_30d"] += float(p["return_30d"])
            if p.get("win_30d"):
                agg[s]["wins_30d"] += 1

    rows = []
    for scanner, d in sorted(agg.items(), key=lambda x: -x[1]["total"]):
        ev7 = d["evaluated_7d"]
        ev30 = d["evaluated_30d"]
        rows.append(
            {
                "Scanner": scanner,
                "Total Picks": d["total"],
                "Win Rate 7d": round(d["wins_7d"] / ev7 * 100, 1) if ev7 else None,
                "Avg Return 7d": round(d["sum_ret_7d"] / ev7, 2) if ev7 else None,
                "Win Rate 30d": round(d["wins_30d"] / ev30 * 100, 1) if ev30 else None,
                "Avg Return 30d": round(d["sum_ret_30d"] / ev30, 2) if ev30 else None,
                "Evaluated 7d": ev7,
            }
        )
    return pd.DataFrame(rows)


def _ranker_lift(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute ranker lift: ranked avg return vs. all-candidates avg return."""
    all_7d, ranked_7d = [], []
    all_30d, ranked_30d = [], []
    for e in events:
        r7 = e.get("return_7d")
        r30 = e.get("return_30d")
        if r7 is not None:
            all_7d.append(float(r7))
            if e.get("was_ranked"):
                ranked_7d.append(float(r7))
        if r30 is not None:
            all_30d.append(float(r30))
            if e.get("was_ranked"):
                ranked_30d.append(float(r30))

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 2) if lst else None

    return {
        "all_7d": safe_mean(all_7d),
        "ranked_7d": safe_mean(ranked_7d),
        "n_all_7d": len(all_7d),
        "n_ranked_7d": len(ranked_7d),
        "all_30d": safe_mean(all_30d),
        "ranked_30d": safe_mean(ranked_30d),
        "n_all_30d": len(all_30d),
        "n_ranked_30d": len(ranked_30d),
        "lift_7d": (
            round(safe_mean(ranked_7d) - safe_mean(all_7d), 2)
            if safe_mean(ranked_7d) is not None and safe_mean(all_7d) is not None
            else None
        ),
        "lift_30d": (
            round(safe_mean(ranked_30d) - safe_mean(all_30d), 2)
            if safe_mean(ranked_30d) is not None and safe_mean(all_30d) is not None
            else None
        ),
    }


def _picks_by_date(picks: List[Dict[str, Any]]) -> pd.DataFrame:
    """Daily pick volume per scanner."""
    rows = defaultdict(lambda: defaultdict(int))
    for p in picks:
        rows[p.get("discovery_date", "?")][p.get("scanner", "unknown")] += 1
    records = []
    for date, scanners in sorted(rows.items()):
        for scanner, count in scanners.items():
            records.append({"Date": date, "Scanner": scanner, "Picks": count})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _kpi(col, label: str, value: str, sub: str = "", color: str = None):
    color = color or COLORS["text_primary"]
    col.markdown(
        f"""
        <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
            border-radius:8px;padding:1rem;">
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em;
                color:{COLORS['text_muted']};margin-bottom:0.4rem;">{label}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;
                font-weight:700;color:{color};">{value}</div>
            <div style="font-size:0.7rem;color:{COLORS['text_muted']};margin-top:0.2rem;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown(
        page_header("Scanners", "Per-scanner win rates & ranker lift"), unsafe_allow_html=True
    )

    picks = load_scanner_picks()
    events = load_discovery_events()

    if not picks:
        st.info("No scanner pick data yet — run the discovery pipeline first.")
        return

    template = get_plotly_template()
    df_metrics = _scanner_metrics(picks)
    lift = _ranker_lift(events)
    df_volume = _picks_by_date(picks)

    # ── KPI row ────────────────────────────────────────────────────────────
    total_picks = len(picks)
    n_scanners = df_metrics["Scanner"].nunique()
    n_days = df_volume["Date"].nunique() if not df_volume.empty else 0
    lift_7d = lift.get("lift_7d")

    cols = st.columns(4)
    _kpi(cols[0], "Total Raw Picks", f"{total_picks:,}", f"{n_days} days")
    _kpi(cols[1], "Active Scanners", str(n_scanners))
    _kpi(cols[2], "Avg Picks / Day", f"{total_picks / n_days:.1f}" if n_days else "—")
    _kpi(
        cols[3],
        "Ranker Lift 7d",
        f"{lift_7d:+.2f}%" if lift_7d is not None else "N/A",
        "accumulating data" if lift_7d is None else "ranked − all candidates",
        color=pnl_color(lift_7d) if lift_7d is not None else COLORS["text_muted"],
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Scanner Table", "📈 Pick Volume", "⚡ Ranker Lift"])

    # ---- Tab 1: per-scanner metrics table --------------------------------
    with tab1:
        st.markdown("##### Per-Scanner Performance")
        st.caption("Win rates accumulate as 7d / 30d forward returns are backfilled nightly.")

        display_df = df_metrics.copy()
        for col in ["Win Rate 7d", "Win Rate 30d"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if x is not None else "—")
        for col in ["Avg Return 7d", "Avg Return 30d"]:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:+.2f}%" if x is not None else "—"
            )
        display_df = display_df.drop(columns=["Evaluated 7d"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Bar chart: total picks per scanner
        fig = go.Figure(
            go.Bar(
                x=df_metrics["Scanner"],
                y=df_metrics["Total Picks"],
                marker_color=COLORS["blue"],
                text=df_metrics["Total Picks"],
                textposition="outside",
            )
        )
        fig.update_layout(**template)
        fig.update_layout(
            title="Total Picks per Scanner",
            xaxis_title="Scanner",
            yaxis_title="Picks",
            height=350,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 2: daily pick volume stacked bar ----------------------------
    with tab2:
        st.markdown("##### Daily Pick Volume by Scanner")
        if df_volume.empty:
            st.info("No data yet.")
        else:
            fig = go.Figure()
            palette = [
                COLORS["blue"],
                COLORS["green"],
                COLORS["amber"],
                COLORS["red"],
                "#a78bfa",
                "#fb7185",
                "#34d399",
                "#fbbf24",
                "#60a5fa",
                "#f472b6",
            ]
            pivot = df_volume.pivot_table(
                index="Date", columns="Scanner", values="Picks", fill_value=0
            )
            for i, scanner in enumerate(pivot.columns):
                fig.add_trace(
                    go.Bar(
                        name=scanner,
                        x=pivot.index.tolist(),
                        y=pivot[scanner].tolist(),
                        marker_color=palette[i % len(palette)],
                    )
                )
            fig.update_layout(**template)
            fig.update_layout(
                barmode="stack",
                title="Picks per Day (stacked by scanner)",
                xaxis_title="Date",
                yaxis_title="Picks",
                height=400,
                margin=dict(t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 3: ranker lift ----------------------------------------------
    with tab3:
        st.markdown("##### Ranker Lift — Does the LLM add value?")
        st.caption(
            "Lift = avg return of ranked picks − avg return of all candidates the ranker saw. "
            "Positive lift means the LLM's selection outperforms random picking from the same pool."
        )

        c1, c2 = st.columns(2)
        for col, horizon, all_key, ranked_key, lift_key, n_all_key, n_ranked_key in [
            (c1, "7d", "all_7d", "ranked_7d", "lift_7d", "n_all_7d", "n_ranked_7d"),
            (c2, "30d", "all_30d", "ranked_30d", "lift_30d", "n_all_30d", "n_ranked_30d"),
        ]:
            all_ret = lift.get(all_key)
            ranked_ret = lift.get(ranked_key)
            lift_val = lift.get(lift_key)
            n_all = lift.get(n_all_key, 0)
            n_ranked = lift.get(n_ranked_key, 0)

            if all_ret is None:
                col.info(f"**{horizon}** — accumulating (need ≥{horizon} of data)")
                continue

            fig = go.Figure()
            labels = [f"All candidates\n(n={n_all})", f"Ranked picks\n(n={n_ranked})"]
            values = [all_ret, ranked_ret if ranked_ret is not None else 0]
            colors = [
                COLORS["text_muted"],
                pnl_color(lift_val) if lift_val is not None else COLORS["amber"],
            ]
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:+.2f}%" for v in values],
                    textposition="outside",
                )
            )
            fig.add_hline(y=0, line_color=COLORS["border"])
            fig.update_layout(**template)
            fig.update_layout(
                title=(
                    f"Avg Return {horizon} — Lift: {lift_val:+.2f}%"
                    if lift_val is not None
                    else f"Avg Return {horizon}"
                ),
                yaxis_title="Avg Return (%)",
                height=320,
                margin=dict(t=50, b=20),
                showlegend=False,
            )
            col.plotly_chart(fig, use_container_width=True)

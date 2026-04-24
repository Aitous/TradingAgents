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
            "evaluated_1d": 0,
            "wins_1d": 0,
            "sum_ret_1d": 0.0,
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
        if p.get("return_1d") is not None:
            agg[s]["evaluated_1d"] += 1
            agg[s]["sum_ret_1d"] += float(p["return_1d"])
            if p.get("win_1d"):
                agg[s]["wins_1d"] += 1
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
        ev1 = d["evaluated_1d"]
        ev7 = d["evaluated_7d"]
        ev30 = d["evaluated_30d"]
        rows.append(
            {
                "Scanner": scanner,
                "Total Picks": d["total"],
                "Win Rate 1d": round(d["wins_1d"] / ev1 * 100, 1) if ev1 else None,
                "Avg Return 1d": round(d["sum_ret_1d"] / ev1, 2) if ev1 else None,
                "Win Rate 7d": round(d["wins_7d"] / ev7 * 100, 1) if ev7 else None,
                "Avg Return 7d": round(d["sum_ret_7d"] / ev7, 2) if ev7 else None,
                "Win Rate 30d": round(d["wins_30d"] / ev30 * 100, 1) if ev30 else None,
                "Avg Return 30d": round(d["sum_ret_30d"] / ev30, 2) if ev30 else None,
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


def _daily_scanner_metrics(picks: List[Dict[str, Any]]) -> pd.DataFrame:
    """Daily per-scanner WR-1d, avg return-1d, and pick count."""
    agg: Dict[tuple, Dict] = defaultdict(
        lambda: {"total": 0, "evaluated": 0, "wins": 0, "sum_ret": 0.0}
    )
    for p in picks:
        key = (p.get("discovery_date", "?"), p.get("scanner", "unknown"))
        agg[key]["total"] += 1
        r1 = p.get("return_1d")
        if r1 is not None:
            agg[key]["evaluated"] += 1
            agg[key]["sum_ret"] += float(r1)
            if p.get("win_1d"):
                agg[key]["wins"] += 1

    rows = []
    for (date, scanner), d in sorted(agg.items()):
        ev = d["evaluated"]
        rows.append({
            "date": date,
            "scanner": scanner,
            "picks": d["total"],
            "wr_1d": round(d["wins"] / ev * 100, 1) if ev else None,
            "avg_ret_1d": round(d["sum_ret"] / ev, 2) if ev else None,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "scanner", "picks", "wr_1d", "avg_ret_1d"]
    )


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
    df_daily = _daily_scanner_metrics(picks)

    if "selected_scanner" not in st.session_state:
        st.session_state["selected_scanner"] = None

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
        st.caption("Click a row to drill down into that scanner's daily trend.")

        display_df = df_metrics.copy()

        def _health(wr):
            if pd.isna(wr):
                return "⬜"
            if wr >= 55:
                return "🟢"
            if wr >= 45:
                return "🟡"
            return "🔴"

        # Health badge: use Win Rate 1d, fall back to Win Rate 7d
        health_wr = display_df["Win Rate 1d"].where(
            display_df["Win Rate 1d"].notna(), display_df["Win Rate 7d"]
        )
        display_df.insert(0, "Health", health_wr.apply(_health))

        # Round numeric columns for display (ProgressColumn needs raw numbers, not strings)
        for col in ["Win Rate 1d", "Win Rate 7d", "Win Rate 30d"]:
            display_df[col] = display_df[col].apply(
                lambda x: round(x, 1) if x is not None else None
            )
        for col in ["Avg Return 1d", "Avg Return 7d", "Avg Return 30d"]:
            display_df[col] = display_df[col].apply(
                lambda x: round(x, 2) if x is not None else None
            )

        # Convert to float so NaN renders as blank instead of "None"
        for col in ["Win Rate 1d", "Win Rate 7d", "Win Rate 30d",
                    "Avg Return 1d", "Avg Return 7d", "Avg Return 30d"]:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")

        selected = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=35 * len(display_df) + 38,  # fit all rows without inner scroll
            column_config={
                "Win Rate 1d": st.column_config.ProgressColumn(
                    "Win Rate 1d",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
                "Win Rate 7d": st.column_config.ProgressColumn(
                    "Win Rate 7d",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
                "Win Rate 30d": st.column_config.ProgressColumn(
                    "Win Rate 30d",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
            },
        )

        # Persist selected scanner in session state
        rows_sel = selected.selection.rows if selected.selection else []
        if rows_sel:
            st.session_state["selected_scanner"] = display_df.iloc[rows_sel[0]]["Scanner"]
        else:
            st.session_state["selected_scanner"] = None

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Drill-down chart ──────────────────────────────────────────────────
        chosen = st.session_state.get("selected_scanner")

        if df_daily.empty or chosen is None:
            st.info("Click a scanner row above to see its daily trend.")
        else:
            scanner_daily = df_daily[df_daily["scanner"] == chosen].copy()
            if scanner_daily.empty:
                st.info(f"No daily data yet for **{chosen}**.")
            else:
                st.markdown(f"##### {chosen} — Daily Trend")

                has_wr = scanner_daily["wr_1d"].notna().any()
                has_ret = scanner_daily["avg_ret_1d"].notna().any()

                fig = go.Figure()

                if has_wr:
                    fig.add_trace(go.Scatter(
                        x=scanner_daily["date"],
                        y=scanner_daily["wr_1d"],
                        name="WR-1d (%)",
                        mode="lines+markers",
                        line=dict(color=COLORS["green"], width=2),
                        yaxis="y1",
                    ))

                if has_ret:
                    fig.add_trace(go.Scatter(
                        x=scanner_daily["date"],
                        y=scanner_daily["avg_ret_1d"],
                        name="Avg Return-1d (%)",
                        mode="lines+markers",
                        line=dict(color=COLORS["blue"], width=2, dash="dot"),
                        yaxis="y1",
                    ))

                fig.add_trace(go.Bar(
                    x=scanner_daily["date"],
                    y=scanner_daily["picks"],
                    name="Picks",
                    marker_color=COLORS["text_muted"],
                    opacity=0.35,
                    yaxis="y2",
                ))

                fig.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color=COLORS["amber"],
                    annotation_text="50% baseline",
                    yref="y1",
                )

                fig.update_layout({
                    **template,
                    "yaxis": dict(title="WR / Avg Return (%)", side="left",
                                  gridcolor="rgba(42, 53, 72, 0.5)", zerolinecolor=COLORS["border"], showgrid=True),
                    "yaxis2": dict(title="Picks", overlaying="y", side="right", showgrid=False),
                    "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    "height": 380,
                    "margin": dict(t=50, b=20),
                })

                if len(scanner_daily) == 1:
                    st.caption("Only 1 day of data — trend builds as more runs complete.")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Total picks bar chart (unchanged) ─────────────────────────────────
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

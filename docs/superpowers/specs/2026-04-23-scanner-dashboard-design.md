# Scanner Dashboard Redesign — Implementation Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade `tradingagents/ui/pages/scanners.py` with color-coded health encoding in the scanner table and a click-to-drill-down trend chart showing WR-1d, avg return-1d, and daily pick count over time.

**Architecture:** All data comes from `data/scanner_picks/*.json` (the new tracking system, live from Apr 20 2026). A new `_daily_scanner_metrics()` aggregation helper groups picks by `(discovery_date, scanner)` and computes daily stats. Selected scanner state lives in `st.session_state["selected_scanner"]`. The drill-down chart updates reactively when a table row is clicked via `st.dataframe(on_select=...)`.

**Tech Stack:** Streamlit, Plotly, Pandas — no new dependencies.

---

## Files

- **Modify:** `tradingagents/ui/pages/scanners.py` — all changes in this one file

---

## Section 1: KPI Row Fix

The "Ranker Lift 7d" card currently overflows with the full string "N/A (accumulating)". Truncate to `"N/A"` with sub-text `"accumulating data"`.

```python
_kpi(
    cols[3],
    "Ranker Lift 7d",
    f"{lift_7d:+.2f}%" if lift_7d is not None else "N/A",
    "accumulating data" if lift_7d is None else "ranked − all candidates",
    color=pnl_color(lift_7d) if lift_7d is not None else COLORS["text_muted"],
)
```

---

## Section 2: New Data Helper — `_daily_scanner_metrics()`

Add this function alongside the existing aggregation helpers. It groups picks by `(discovery_date, scanner)` and returns a DataFrame with one row per (date, scanner).

```python
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
```

---

## Section 3: Health Encoding in Scanner Table

Replace the `tab1` scanner table block. Add a `Health` column and color-code `Win Rate 7d`.

**Health badge logic:**
- `🟢` if `Win Rate 7d >= 55`
- `🟡` if `45 <= Win Rate 7d < 55`
- `🔴` if `Win Rate 7d < 45`
- `⬜` if no data yet

**Color logic for WR cells:**
Use `st.dataframe` with `column_config` to apply background gradients. The `Win Rate 7d` column uses a `ProgressColumn` styled 0–100 range so high values appear green and low values appear red.

```python
with tab1:
    st.markdown("##### Per-Scanner Performance")
    st.caption("Click a row to drill down into that scanner's daily trend.")

    display_df = df_metrics.copy()

    # Add health badge
    def _health(wr):
        if wr is None:
            return "⬜"
        if wr >= 55:
            return "🟢"
        if wr >= 45:
            return "🟡"
        return "🔴"

    display_df.insert(0, "Health", display_df["Win Rate 7d"].apply(_health))

    for col in ["Win Rate 7d", "Win Rate 30d"]:
        display_df[col] = display_df[col].apply(
            lambda x: round(x, 1) if x is not None else None
        )
    for col in ["Avg Return 7d", "Avg Return 30d"]:
        display_df[col] = display_df[col].apply(
            lambda x: round(x, 2) if x is not None else None
        )
    display_df = display_df.drop(columns=["Evaluated 7d"])

    selected = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
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
    rows_sel = selected.selection.get("rows", []) if selected else []
    if rows_sel:
        st.session_state["selected_scanner"] = display_df.iloc[rows_sel[0]]["Scanner"]

    # ── Drill-down chart ───────────────────────────────────────────────────
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

            # WR-1d line
            if has_wr:
                fig.add_trace(go.Scatter(
                    x=scanner_daily["date"],
                    y=scanner_daily["wr_1d"],
                    name="WR-1d (%)",
                    mode="lines+markers",
                    line=dict(color=COLORS["green"], width=2),
                    yaxis="y1",
                ))

            # Avg return-1d line
            if has_ret:
                fig.add_trace(go.Scatter(
                    x=scanner_daily["date"],
                    y=scanner_daily["avg_ret_1d"],
                    name="Avg Return-1d (%)",
                    mode="lines+markers",
                    line=dict(color=COLORS["blue"], width=2, dash="dot"),
                    yaxis="y1",
                ))

            # Pick count bars on secondary axis
            fig.add_trace(go.Bar(
                x=scanner_daily["date"],
                y=scanner_daily["picks"],
                name="Picks",
                marker_color=COLORS["text_muted"],
                opacity=0.35,
                yaxis="y2",
            ))

            # 50% WR baseline
            fig.add_hline(y=50, line_dash="dash", line_color=COLORS["amber"],
                          annotation_text="50% baseline", yref="y1")

            fig.update_layout(
                **template,
                yaxis=dict(title="WR / Avg Return (%)", side="left"),
                yaxis2=dict(title="Picks", overlaying="y", side="right",
                            showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                height=380,
                margin=dict(t=50, b=20),
            )
            if len(scanner_daily) == 1:
                st.caption("Only 1 day of data — trend builds as more runs complete.")
            st.plotly_chart(fig, use_container_width=True)

    # Total picks bar chart (unchanged)
    fig = go.Figure(go.Bar(...))  # keep existing picks bar chart below
```

> **Note to implementer:** Keep the existing "Total picks per scanner" bar chart at the bottom of tab1, below the drill-down section. Do not remove it.

---

## Section 4: Wire `df_daily` into `render()`

In `render()`, compute `df_daily` alongside the existing aggregations and initialize `session_state`:

```python
df_daily = _daily_scanner_metrics(picks)

if "selected_scanner" not in st.session_state:
    st.session_state["selected_scanner"] = None
```

Pass `df_daily` and `template` into the tab1 block (they are already in scope in `render()`).

---

## Success Criteria

1. Table shows `Health` column with correct colored dot for each scanner
2. `Win Rate 7d` column renders as a progress bar (green = high, red = low via Streamlit ProgressColumn)
3. Clicking a scanner row updates the drill-down chart below
4. Drill-down shows three traces: WR-1d (green line), avg return-1d (blue dashed), picks (grey bars)
5. 50% baseline reference line visible on chart
6. KPI "Ranker Lift 7d" card shows "N/A" without overflow
7. Single-day data case shows a point + caption instead of an empty chart
8. All other tabs (Pick Volume, Ranker Lift) unchanged

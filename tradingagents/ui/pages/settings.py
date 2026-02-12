"""
Config page — displays pipeline configuration in a terminal-style layout.

Read-only view of scanners, pipelines, and data source configuration.
"""

import streamlit as st

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.ui.theme import COLORS, page_header


def render() -> None:
    """Render the configuration page."""
    st.markdown(
        page_header("Config", "Pipeline & scanner configuration (read-only)"),
        unsafe_allow_html=True,
    )

    config = DEFAULT_CONFIG
    discovery_config = config.get("discovery", {})

    # ---- Top-level settings ----
    st.markdown(
        '<div class="section-title">Discovery Settings <span class="accent">// core</span></div>',
        unsafe_allow_html=True,
    )

    settings_grid = [
        ("Discovery Mode", discovery_config.get("discovery_mode", "N/A")),
        ("Max Candidates", str(discovery_config.get("max_candidates_to_analyze", "N/A"))),
        ("Final Recommendations", str(discovery_config.get("final_recommendations", "N/A"))),
        ("Deep Dive Workers", str(discovery_config.get("deep_dive_max_workers", "N/A"))),
    ]

    cols = st.columns(len(settings_grid))
    for col, (label, val) in zip(cols, settings_grid):
        with col:
            st.markdown(
                f"""
                <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
                    border-radius:8px;padding:0.75rem 1rem;text-align:center;">
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.6rem;
                        font-weight:600;text-transform:uppercase;letter-spacing:0.06em;
                        color:{COLORS['text_muted']};margin-bottom:0.3rem;">{label}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                        font-weight:600;color:{COLORS['text_primary']};">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ---- Pipelines ----
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(
            '<div class="section-title">Pipelines <span class="accent">// routing</span></div>',
            unsafe_allow_html=True,
        )

        pipelines = discovery_config.get("pipelines", {})
        for name, cfg in pipelines.items():
            enabled = cfg.get("enabled", False)
            priority = cfg.get("priority", "N/A")
            budget = cfg.get("deep_dive_budget", "N/A")
            status_color = COLORS["green"] if enabled else COLORS["red"]
            status_dot = f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{status_color};margin-right:0.4rem;vertical-align:middle;"></span>'

            st.markdown(
                f"""
                <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
                    border-radius:8px;padding:0.65rem 0.85rem;margin-bottom:0.4rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                            font-weight:600;color:{COLORS['text_primary']};">
                            {status_dot}{name}
                        </span>
                        <div style="display:flex;gap:0.75rem;">
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                color:{COLORS['text_muted']};">P:{priority}</span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                color:{COLORS['text_muted']};">B:{budget}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown(
            '<div class="section-title">Scanners <span class="accent">// sources</span></div>',
            unsafe_allow_html=True,
        )

        scanners = discovery_config.get("scanners", {})
        for name, cfg in scanners.items():
            enabled = cfg.get("enabled", False)
            pipeline = cfg.get("pipeline", "N/A")
            limit = cfg.get("limit", "N/A")
            status_color = COLORS["green"] if enabled else COLORS["red"]
            status_dot = f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{status_color};margin-right:0.4rem;vertical-align:middle;"></span>'

            st.markdown(
                f"""
                <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
                    border-radius:8px;padding:0.55rem 0.85rem;margin-bottom:0.35rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                            font-weight:500;color:{COLORS['text_primary']};">
                            {status_dot}{name.replace('_', ' ')}
                        </span>
                        <div style="display:flex;gap:0.75rem;">
                            <span style="font-family:'DM Sans',sans-serif;font-size:0.6rem;
                                font-weight:600;text-transform:uppercase;
                                padding:0.1rem 0.35rem;border-radius:3px;
                                background:rgba(59,130,246,0.12);
                                color:{COLORS['blue']};letter-spacing:0.04em;">{pipeline}</span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                color:{COLORS['text_muted']};">limit:{limit}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- Data Sources ----
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Data Sources <span class="accent">// vendors</span></div>',
        unsafe_allow_html=True,
    )

    data_vendors = config.get("data_vendors", {})
    if data_vendors:
        cols = st.columns(3)
        for i, (vendor_type, vendor_name) in enumerate(data_vendors.items()):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
                        border-radius:6px;padding:0.5rem 0.75rem;margin-bottom:0.35rem;">
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.6rem;
                            color:{COLORS['text_muted']};text-transform:uppercase;
                            letter-spacing:0.04em;">{vendor_type.replace('_', ' ')}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                            color:{COLORS['text_primary']};font-weight:500;margin-top:0.15rem;">
                            {vendor_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No data sources configured.")

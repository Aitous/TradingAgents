"""
Main Streamlit app entry point for the Trading Agents Dashboard.

Dark terminal-inspired trading interface with sidebar navigation.
"""

from datetime import datetime

import streamlit as st

from tradingagents.ui import pages
from tradingagents.ui.theme import COLORS, GLOBAL_CSS
from tradingagents.ui.utils import load_quick_stats, load_recommendations, load_scanner_health


def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Trading Agents",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar():
    """Render the sidebar with navigation and quick stats."""
    with st.sidebar:
        # Brand header
        st.markdown(
            f"""
            <div style="padding:0.5rem 0 1.25rem 0;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.15rem;
                    font-weight:700;color:{COLORS['text_primary']};letter-spacing:-0.03em;">
                    TRADING<span style="color:{COLORS['green']};">AGENTS</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                    color:{COLORS['text_muted']};margin-top:0.15rem;">
                    v2.0 &mdash; {datetime.now().strftime('%b %d, %Y')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div style="height:1px;background:{COLORS['border']};
                margin-bottom:1rem;"></div>""",
            unsafe_allow_html=True,
        )

        # Navigation
        page = st.radio(
            "Navigation",
            options=[
                "Overview",
                "Signals",
                "Portfolio",
                "Performance",
                "Scanners",
                "Hypotheses",
                "Config",
            ],
            label_visibility="collapsed",
        )

        st.markdown(
            f"""<div style="height:1px;background:{COLORS['border']};
                margin:1rem 0;"></div>""",
            unsafe_allow_html=True,
        )

        # System pulse
        try:
            open_positions, win_rate = load_quick_stats()
            recs = load_recommendations()
            n_signals = len(recs) if recs else 0
            top_ticker = ""
            if recs:
                ranked = sorted(
                    recs,
                    key=lambda r: (r.get("final_score", 0) or 0) * (r.get("confidence", 0) or 0),
                    reverse=True,
                )
                top_ticker = ranked[0].get("ticker", "") if ranked else ""

            scanner_health = load_scanner_health()
            healthy = sum(1 for s in scanner_health if s["health"] == "green")
            total_scanners = len(scanner_health)
            scanner_ratio = f"{healthy}/{total_scanners}" if total_scanners else "—"
            wr_color = COLORS["green"] if win_rate >= 50 else COLORS["red"]

            top_line = (
                f'<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:0.1rem;">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;'
                f'font-weight:700;color:{COLORS["text_primary"]};">{top_ticker}</span>'
                f'<span style="font-size:0.6rem;color:{COLORS["text_muted"]};">top pick</span>'
                f"</div>"
                if top_ticker
                else ""
            )

            st.markdown(
                f"""
                <div style="padding:0.75rem;background:{COLORS['bg_card']};
                    border:1px solid {COLORS['border']};border-radius:8px;">
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.6rem;
                        font-weight:600;text-transform:uppercase;letter-spacing:0.08em;
                        color:{COLORS['text_muted']};margin-bottom:0.6rem;">System Pulse</div>
                    {top_line}
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem;">
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;
                                font-size:1.1rem;font-weight:700;color:{wr_color};">
                                {win_rate:.0f}%
                            </div>
                            <div style="font-size:0.6rem;color:{COLORS['text_muted']};">Win Rate 7d</div>
                        </div>
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;
                                font-size:1.1rem;font-weight:700;color:{COLORS['text_primary']};">
                                {n_signals}
                            </div>
                            <div style="font-size:0.6rem;color:{COLORS['text_muted']};">Signals today</div>
                        </div>
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;
                                font-size:1.1rem;font-weight:700;color:{COLORS['text_primary']};">
                                {open_positions}
                            </div>
                            <div style="font-size:0.6rem;color:{COLORS['text_muted']};">Open positions</div>
                        </div>
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;
                                font-size:1.1rem;font-weight:700;
                                color:{COLORS['green'] if healthy == total_scanners and total_scanners > 0 else COLORS['amber']};">
                                {scanner_ratio}
                            </div>
                            <div style="font-size:0.6rem;color:{COLORS['text_muted']};">Scanners green</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        return page


def route_page(page):
    """Route to the appropriate page based on selection."""
    page_map = {
        "Overview": pages.home,
        "Signals": pages.todays_picks,
        "Portfolio": pages.portfolio,
        "Performance": pages.performance,
        "Scanners": pages.scanners,
        "Hypotheses": pages.hypotheses,
        "Config": pages.settings,
    }
    module = page_map.get(page)
    if module is None:
        st.error(f"Unknown page: {page}")
        return
    try:
        module.render()
    except Exception as exc:
        st.error(f"Error rendering {page}: {exc}")
        import traceback

        st.code(traceback.format_exc(), language="python")


def main():
    """Main entry point for the Streamlit app."""
    setup_page_config()

    # Inject global theme CSS
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Route to selected page
    route_page(selected_page)


if __name__ == "__main__":
    main()

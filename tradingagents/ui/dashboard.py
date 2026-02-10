"""
Main Streamlit app entry point for the Trading Agents Dashboard.

This module sets up the dashboard page configuration, sidebar navigation,
and routing to different pages based on user selection.
"""

import streamlit as st

from tradingagents.ui import pages
from tradingagents.ui.utils import load_quick_stats


def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Trading Agents Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar():
    """Render the sidebar with navigation and quick stats."""
    with st.sidebar:
        st.title("Trading Agents")

        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select a page:",
            options=["Home", "Today's Picks", "Portfolio", "Performance", "Settings"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Quick stats section
        st.markdown("### Quick Stats")
        try:
            open_positions, win_rate = load_quick_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Open Positions", open_positions)
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%")
        except Exception as e:
            st.warning(f"Could not load quick stats: {str(e)}")

        return page


def route_page(page):
    """Route to the appropriate page based on selection."""
    if page == "Home":
        pages.home.render()
    elif page == "Today's Picks":
        pages.todays_picks.render()
    elif page == "Portfolio":
        pages.portfolio.render()
    elif page == "Performance":
        pages.performance.render()
    elif page == "Settings":
        pages.settings.render()
    else:
        st.error(f"Unknown page: {page}")


def main():
    """Main entry point for the Streamlit app."""
    setup_page_config()

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    .main {
        padding: 2rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Route to selected page
    route_page(selected_page)


if __name__ == "__main__":
    main()

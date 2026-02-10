"""
Settings page for the Trading Agents Dashboard.

This module displays configuration settings and scanner/pipeline status information.
It provides a read-only view of current settings with expandable sections for detailed configuration.
"""

import streamlit as st

from tradingagents.default_config import DEFAULT_CONFIG


def render() -> None:
    """
    Render the settings page.

    Displays:
    - Page title
    - Configuration info message
    - Discovery configuration settings
    - Pipelines section with expandable cards showing:
      - enabled status
      - priority
      - deep_dive_budget
    - Scanners section with checkboxes showing:
      - enabled status for each scanner
    """
    # Page title
    st.title("⚙️ Settings")

    # Info message
    st.info("Configuration UI - TODO: Implement save functionality")

    # Get configuration
    config = DEFAULT_CONFIG
    discovery_config = config.get("discovery", {})

    # Display current configuration section
    st.subheader("📋 Configuration")

    # Show key discovery settings
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Discovery Mode",
            value=discovery_config.get("discovery_mode", "N/A"),
        )

    with col2:
        st.metric(
            label="Max Candidates",
            value=discovery_config.get("max_candidates_to_analyze", "N/A"),
        )

    with col3:
        st.metric(
            label="Final Recommendations",
            value=discovery_config.get("final_recommendations", "N/A"),
        )

    # Pipelines section
    st.subheader("🔄 Pipelines")

    pipelines = discovery_config.get("pipelines", {})

    if pipelines:
        for pipeline_name, pipeline_config in pipelines.items():
            with st.expander(
                f"{'✅' if pipeline_config.get('enabled') else '❌'} {pipeline_name.title()}"
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Enabled",
                        value="Yes" if pipeline_config.get("enabled") else "No",
                    )

                with col2:
                    st.metric(
                        label="Priority",
                        value=pipeline_config.get("priority", "N/A"),
                    )

                with col3:
                    st.metric(
                        label="Budget",
                        value=pipeline_config.get("deep_dive_budget", "N/A"),
                    )

                if "ranker_prompt" in pipeline_config:
                    st.caption(f"Ranker: {pipeline_config.get('ranker_prompt', 'N/A')}")
    else:
        st.info("No pipelines configured")

    # Scanners section
    st.subheader("🔍 Scanners")

    scanners = discovery_config.get("scanners", {})

    if scanners:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**Scanner Status**")

        with col2:
            st.write("**Enabled**")

        # Display each scanner with checkbox showing enabled status
        for scanner_name, scanner_config in scanners.items():
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"• {scanner_name.replace('_', ' ').title()}")

            with col2:
                is_enabled = scanner_config.get("enabled", False)
                st.write("✅" if is_enabled else "❌")

        # Additional scanner configuration in expander
        with st.expander("📊 Scanner Details"):
            for scanner_name, scanner_config in scanners.items():
                pipeline = scanner_config.get("pipeline", "N/A")
                limit = scanner_config.get("limit", "N/A")
                enabled = scanner_config.get("enabled", False)

                st.write(
                    f"**{scanner_name}** | "
                    f"Pipeline: {pipeline} | "
                    f"Limit: {limit} | "
                    f"Status: {'✅ Enabled' if enabled else '❌ Disabled'}"
                )
    else:
        st.info("No scanners configured")

    # Data sources section
    st.subheader("📡 Data Sources")

    data_vendors = config.get("data_vendors", {})

    if data_vendors:
        for vendor_type, vendor_name in data_vendors.items():
            st.write(f"**{vendor_type.replace('_', ' ').title()}**: {vendor_name}")
    else:
        st.info("No data sources configured")

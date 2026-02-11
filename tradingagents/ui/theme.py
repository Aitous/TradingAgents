"""
Trading terminal dark theme for the Streamlit dashboard.

Bloomberg/TradingView-inspired aesthetic with green/amber accents.
Uses CSS variables for consistency and injects custom fonts.
"""

# -- Color Tokens --
COLORS = {
    "bg_primary": "#0a0e17",
    "bg_secondary": "#111827",
    "bg_card": "#1a2234",
    "bg_card_hover": "#1f2b42",
    "bg_input": "#151d2e",
    "border": "#2a3548",
    "border_active": "#3b82f6",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "green": "#22c55e",
    "green_dim": "#16a34a",
    "green_glow": "rgba(34, 197, 94, 0.15)",
    "red": "#ef4444",
    "red_dim": "#dc2626",
    "red_glow": "rgba(239, 68, 68, 0.15)",
    "amber": "#f59e0b",
    "amber_dim": "#d97706",
    "blue": "#3b82f6",
    "blue_dim": "#2563eb",
    "cyan": "#06b6d4",
    "purple": "#a855f7",
}


def get_plotly_template():
    """Return a Plotly layout template matching the terminal theme."""
    return dict(
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_card"],
        font=dict(
            family="JetBrains Mono, SF Mono, Menlo, monospace",
            color=COLORS["text_secondary"],
            size=11,
        ),
        xaxis=dict(
            gridcolor="rgba(42, 53, 72, 0.5)",
            zerolinecolor=COLORS["border"],
            showgrid=True,
            gridwidth=1,
        ),
        yaxis=dict(
            gridcolor="rgba(42, 53, 72, 0.5)",
            zerolinecolor=COLORS["border"],
            showgrid=True,
            gridwidth=1,
        ),
        margin=dict(l=0, r=0, t=32, b=0),
        hoverlabel=dict(
            bgcolor=COLORS["bg_secondary"],
            font_color=COLORS["text_primary"],
            bordercolor=COLORS["border"],
        ),
        colorway=[
            COLORS["green"],
            COLORS["blue"],
            COLORS["amber"],
            COLORS["cyan"],
            COLORS["purple"],
            COLORS["red"],
        ],
    )


GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

/* ---- Root overrides ---- */
:root {{
    --bg-primary: {COLORS["bg_primary"]};
    --bg-secondary: {COLORS["bg_secondary"]};
    --bg-card: {COLORS["bg_card"]};
    --border: {COLORS["border"]};
    --text-primary: {COLORS["text_primary"]};
    --text-secondary: {COLORS["text_secondary"]};
    --green: {COLORS["green"]};
    --red: {COLORS["red"]};
    --amber: {COLORS["amber"]};
    --blue: {COLORS["blue"]};
}}

/* ---- Global ---- */
.stApp {{
    background-color: var(--bg-primary) !important;
    color: var(--text-primary);
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* Hide default Streamlit chrome */
header[data-testid="stHeader"] {{
    background-color: var(--bg-primary) !important;
    border-bottom: 1px solid var(--border);
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] .stRadio label {{
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    transition: all 0.15s ease;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {{
    background-color: var(--bg-card) !important;
    color: var(--green) !important;
    border-left: 3px solid var(--green);
}}

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
}}
div[data-testid="stMetric"] label {{
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg {{
    display: none;
}}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] > div {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}}

/* ---- Custom KPI card classes ---- */
.kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 10px 10px 0 0;
}}
.kpi-card.green::before {{ background: var(--green); }}
.kpi-card.red::before {{ background: var(--red); }}
.kpi-card.amber::before {{ background: var(--amber); }}
.kpi-card.blue::before {{ background: var(--blue); }}
.kpi-label {{
    color: var(--text-secondary);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}}
.kpi-value {{
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
}}
.kpi-delta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.25rem;
}}
.kpi-delta.positive {{ color: var(--green); }}
.kpi-delta.negative {{ color: var(--red); }}
.kpi-delta.neutral {{ color: var(--text-muted); }}

/* ---- Signal card (recommendation) ---- */
.signal-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
.signal-card:hover {{
    border-color: {COLORS["border_active"]};
    box-shadow: 0 0 0 1px {COLORS["border_active"]};
}}
.signal-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}}
.signal-ticker {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
}}
.signal-rank {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    background: var(--bg-secondary);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
}}
.signal-badges {{
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
}}
.badge {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.badge-green {{
    background: {COLORS["green_glow"]};
    color: var(--green);
    border: 1px solid {COLORS["green_dim"]};
}}
.badge-red {{
    background: {COLORS["red_glow"]};
    color: var(--red);
    border: 1px solid {COLORS["red_dim"]};
}}
.badge-amber {{
    background: rgba(245, 158, 11, 0.15);
    color: var(--amber);
    border: 1px solid {COLORS["amber_dim"]};
}}
.badge-blue {{
    background: rgba(59, 130, 246, 0.15);
    color: var(--blue);
    border: 1px solid {COLORS["blue_dim"]};
}}
.badge-muted {{
    background: rgba(100, 116, 139, 0.15);
    color: var(--text-secondary);
    border: 1px solid {COLORS["border"]};
}}
.signal-metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}}
.signal-metric {{
    text-align: center;
}}
.signal-metric-label {{
    font-size: 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
}}
.signal-metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 0.15rem;
}}
.signal-thesis {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    line-height: 1.55;
    color: var(--text-secondary);
    padding: 0.75rem;
    background: var(--bg-secondary);
    border-radius: 6px;
    border-left: 3px solid var(--border);
}}

/* ---- Confidence bar ---- */
.conf-bar {{
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 0.5rem;
}}
.conf-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
}}

/* ---- Strategy tag colors ---- */
.strat-momentum {{ border-left-color: var(--green) !important; }}
.strat-insider {{ border-left-color: var(--amber) !important; }}
.strat-earnings {{ border-left-color: var(--blue) !important; }}
.strat-volume {{ border-left-color: {COLORS["cyan"]} !important; }}
.strat-options {{ border-left-color: {COLORS["purple"]} !important; }}

/* ---- Table styling ---- */
.stDataFrame {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border);
    border-radius: 8px;
}}
div[data-testid="stDataFrame"] table {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}}
div[data-testid="stDataFrame"] th {{
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em;
}}

/* ---- Expander ---- */
.streamlit-expanderHeader {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600;
    color: var(--text-primary) !important;
}}

/* ---- Buttons ---- */
.stButton > button {{
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-primary);
    transition: all 0.15s ease;
}}
.stButton > button:hover {{
    border-color: var(--green);
    color: var(--green);
    background: {COLORS["green_glow"]};
}}

/* ---- Divider ---- */
hr {{
    border-color: var(--border) !important;
    opacity: 0.5;
}}

/* ---- Subheader ---- */
.stMarkdown h2, .stMarkdown h3 {{
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}}

/* ---- Selectbox & slider ---- */
div[data-baseweb="select"] {{
    font-family: 'JetBrains Mono', monospace;
}}

/* ---- Scrollbar ---- */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: var(--bg-primary);
}}
::-webkit-scrollbar-thumb {{
    background: var(--border);
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {COLORS["text_muted"]};
}}

/* ---- Section title with mono accent ---- */
.section-title {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.section-title .accent {{
    font-family: 'JetBrains Mono', monospace;
    color: var(--green);
    font-size: 0.85rem;
}}

/* ---- Page header ---- */
.page-header {{
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}}
.page-header h1 {{
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.02em;
}}
.page-header .subtitle {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}}

/* ---- Position row ---- */
.pos-row {{
    display: grid;
    grid-template-columns: 80px 1fr repeat(5, 90px);
    align-items: center;
    padding: 0.75rem 1rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}}
.pos-ticker {{
    font-weight: 700;
    color: var(--text-primary);
    font-size: 0.95rem;
}}

</style>
"""


def kpi_card(label: str, value: str, delta: str = "", color: str = "blue") -> str:
    """Render a custom KPI card as HTML."""
    delta_class = (
        "positive"
        if delta.startswith("+")
        else ("negative" if delta.startswith("-") else "neutral")
    )
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card {color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def page_header(title: str, subtitle: str = "") -> str:
    """Render a page header as HTML."""
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="page-header">
        <h1>{title}</h1>
        {sub}
    </div>
    """


def signal_card(
    rank: int,
    ticker: str,
    score: int,
    confidence: int,
    strategy: str,
    entry_price: float,
    reason: str,
) -> str:
    """Render a recommendation signal card as HTML."""
    # Confidence bar color
    if confidence >= 8:
        bar_color = COLORS["green"]
    elif confidence >= 6:
        bar_color = COLORS["amber"]
    else:
        bar_color = COLORS["red"]

    # Score badge color
    if score >= 40:
        score_badge = "badge-green"
    elif score >= 25:
        score_badge = "badge-amber"
    else:
        score_badge = "badge-muted"

    # Strategy badge
    strat_badge = "badge-blue"
    strat_css = ""
    strat_lower = strategy.lower().replace(" ", "_")
    if "momentum" in strat_lower:
        strat_badge = "badge-green"
        strat_css = "strat-momentum"
    elif "insider" in strat_lower:
        strat_badge = "badge-amber"
        strat_css = "strat-insider"
    elif "earnings" in strat_lower:
        strat_badge = "badge-blue"
        strat_css = "strat-earnings"
    elif "volume" in strat_lower:
        strat_badge = "badge-blue"
        strat_css = "strat-volume"

    entry_str = f"${entry_price:.2f}" if entry_price else "N/A"
    conf_pct = confidence * 10

    return f"""
    <div class="signal-card {strat_css}">
        <div class="signal-header">
            <div style="display:flex;align-items:center;gap:0.75rem;">
                <span class="signal-ticker">{ticker}</span>
                <span class="signal-rank">#{rank}</span>
            </div>
        </div>
        <div class="signal-badges">
            <span class="badge {strat_badge}">{strategy}</span>
            <span class="badge {score_badge}">Score {score}</span>
            <span class="badge badge-muted">Conf {confidence}/10</span>
        </div>
        <div class="signal-metrics">
            <div class="signal-metric">
                <div class="signal-metric-label">Entry</div>
                <div class="signal-metric-value">{entry_str}</div>
            </div>
            <div class="signal-metric">
                <div class="signal-metric-label">Score</div>
                <div class="signal-metric-value">{score}</div>
            </div>
            <div class="signal-metric">
                <div class="signal-metric-label">Confidence</div>
                <div class="signal-metric-value">{confidence}/10</div>
            </div>
            <div class="signal-metric">
                <div class="signal-metric-label">Strategy</div>
                <div class="signal-metric-value" style="font-size:0.75rem;">{strategy.upper()}</div>
            </div>
        </div>
        <div class="signal-thesis">{reason}</div>
        <div class="conf-bar">
            <div class="conf-fill" style="width:{conf_pct}%;background:{bar_color};"></div>
        </div>
    </div>
    """


def pnl_color(value: float) -> str:
    """Return green/red CSS color based on sign."""
    if value > 0:
        return COLORS["green"]
    elif value < 0:
        return COLORS["red"]
    return COLORS["text_muted"]

"""
Dashboard page modules for the Trading Agents UI.

This package contains all page modules that can be rendered in the dashboard.
Each module should have a render() function that displays the page content.
"""

try:
    from tradingagents.ui.pages import home
except ImportError:
    home = None

try:
    from tradingagents.ui.pages import todays_picks
except ImportError:
    todays_picks = None

try:
    from tradingagents.ui.pages import portfolio
except ImportError:
    portfolio = None

try:
    from tradingagents.ui.pages import performance
except ImportError:
    performance = None

try:
    from tradingagents.ui.pages import settings
except ImportError:
    settings = None


__all__ = [
    "home",
    "todays_picks",
    "portfolio",
    "performance",
    "settings",
]

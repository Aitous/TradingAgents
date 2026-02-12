# Modular Multi-Pipeline Discovery Architecture - Fast Implementation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform discovery system into modular, multi-pipeline architecture with early signal scanners, dynamic performance tracking, and Streamlit dashboard UI.

**Approach:** Implementation-first, skip tests/docs for fast experimentation.

**Branch:** `feature/modular-pipeline-architecture` (no git commits during implementation)

---

## Phase 1: Core Architecture (30 min)

### Task 1: Create Scanner Registry

**Files:**
- Create: `tradingagents/dataflows/discovery/scanner_registry.py`

**Implementation:**

```python
# tradingagents/dataflows/discovery/scanner_registry.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type


class BaseScanner(ABC):
    """Base class for all discovery scanners."""

    name: str = None
    pipeline: str = None

    def __init__(self, config: Dict[str, Any]):
        if self.name is None:
            raise ValueError(f"{self.__class__.__name__} must define 'name'")
        if self.pipeline is None:
            raise ValueError(f"{self.__class__.__name__} must define 'pipeline'")

        self.config = config
        self.scanner_config = config.get("discovery", {}).get("scanners", {}).get(self.name, {})
        self.enabled = self.scanner_config.get("enabled", True)
        self.limit = self.scanner_config.get("limit", 10)

    @abstractmethod
    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return list of candidates with: ticker, source, context, priority"""
        pass

    def is_enabled(self) -> bool:
        return self.enabled


class ScannerRegistry:
    """Global scanner registry."""

    def __init__(self):
        self.scanners: Dict[str, Type[BaseScanner]] = {}

    def register(self, scanner_class: Type[BaseScanner]):
        if not hasattr(scanner_class, "name") or scanner_class.name is None:
            raise ValueError(f"Scanner must define 'name'")
        if not hasattr(scanner_class, "pipeline") or scanner_class.pipeline is None:
            raise ValueError(f"Scanner must define 'pipeline'")
        self.scanners[scanner_class.name] = scanner_class

    def get_scanners_by_pipeline(self, pipeline: str) -> List[Type[BaseScanner]]:
        return [sc for sc in self.scanners.values() if sc.pipeline == pipeline]

    def get_all_scanners(self) -> List[Type[BaseScanner]]:
        return list(self.scanners.values())


SCANNER_REGISTRY = ScannerRegistry()
```

---

### Task 2: Update Config with Modular Structure

**Files:**
- Modify: `tradingagents/default_config.py`

**Add to config:**

```python
"discovery": {
    # ... existing settings ...

    # PIPELINES: Define ranking behavior per pipeline
    "pipelines": {
        "edge": {
            "enabled": True,
            "priority": 1,
            "ranker_prompt": "edge_signals_ranker.txt",
            "deep_dive_budget": 15
        },
        "momentum": {
            "enabled": True,
            "priority": 2,
            "ranker_prompt": "momentum_ranker.txt",
            "deep_dive_budget": 10
        },
        "news": {
            "enabled": True,
            "priority": 3,
            "ranker_prompt": "news_catalyst_ranker.txt",
            "deep_dive_budget": 5
        },
        "social": {
            "enabled": True,
            "priority": 4,
            "ranker_prompt": "social_signals_ranker.txt",
            "deep_dive_budget": 5
        },
        "events": {
            "enabled": False,
            "priority": 5,
            "deep_dive_budget": 0
        }
    },

    # SCANNERS: Each declares its pipeline
    "scanners": {
        # Edge signals
        "insider_buying": {"enabled": True, "pipeline": "edge", "limit": 20},
        "options_flow": {"enabled": True, "pipeline": "edge", "limit": 15},
        "congress_trades": {"enabled": False, "pipeline": "edge", "limit": 10},

        # Momentum
        "volume_accumulation": {"enabled": True, "pipeline": "momentum", "limit": 15},
        "market_movers": {"enabled": True, "pipeline": "momentum", "limit": 10},

        # News
        "semantic_news": {"enabled": True, "pipeline": "news", "limit": 10},
        "analyst_upgrade": {"enabled": False, "pipeline": "news", "limit": 5},

        # Social
        "reddit_trending": {"enabled": True, "pipeline": "social", "limit": 15},
        "reddit_dd": {"enabled": True, "pipeline": "social", "limit": 10},

        # Events
        "earnings_calendar": {"enabled": False, "pipeline": "events", "limit": 10},
        "short_squeeze": {"enabled": False, "pipeline": "events", "limit": 5}
    }
}
```

---

## Phase 2: New Edge Scanners (45 min)

### Task 3: Insider Buying Scanner

**Files:**
- Create: `tradingagents/dataflows/discovery/scanners/insider_buying.py`

**Implementation:**

```python
# tradingagents/dataflows/discovery/scanners/insider_buying.py
"""SEC Form 4 insider buying scanner."""
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import BaseScanner, SCANNER_REGISTRY


class InsiderBuyingScanner(BaseScanner):
    """Scan SEC Form 4 for insider purchases."""

    name = "insider_buying"
    pipeline = "edge"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_days = self.scanner_config.get("lookback_days", 7)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        print(f"   💼 Scanning insider buying (last {self.lookback_days} days)...")

        try:
            # Use existing FMP API or placeholder
            # For MVP: Return empty or use FMP insider trades endpoint
            candidates = []

            # TODO: Implement actual Form 4 fetching
            # For now, placeholder that uses FMP API if available

            print(f"      Found {len(candidates)} insider purchases")
            return candidates

        except Exception as e:
            print(f"      Error: {e}")
            return []


SCANNER_REGISTRY.register(InsiderBuyingScanner)
```

---

### Task 4: Options Flow Scanner

**Files:**
- Create: `tradingagents/dataflows/discovery/scanners/options_flow.py`

**Implementation:**

```python
# tradingagents/dataflows/discovery/scanners/options_flow.py
"""Unusual options activity scanner."""
from typing import Any, Dict, List
import yfinance as yf

from tradingagents.dataflows.discovery.scanner_registry import BaseScanner, SCANNER_REGISTRY


class OptionsFlowScanner(BaseScanner):
    """Scan for unusual options activity."""

    name = "options_flow"
    pipeline = "edge"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_volume_oi_ratio = self.scanner_config.get("min_volume_oi_ratio", 2.0)
        # Focus on liquid options
        self.ticker_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA"]

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        print(f"   📊 Scanning unusual options activity...")

        candidates = []

        for ticker in self.ticker_universe[:20]:  # Limit for speed
            try:
                unusual = self._analyze_ticker_options(ticker)
                if unusual:
                    candidates.append(unusual)
                if len(candidates) >= self.limit:
                    break
            except:
                continue

        print(f"      Found {len(candidates)} unusual options flows")
        return candidates

    def _analyze_ticker_options(self, ticker: str) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                return None

            options = stock.option_chain(expirations[0])
            calls = options.calls
            puts = options.puts

            # Find unusual strikes
            unusual_strikes = []
            for _, opt in calls.iterrows():
                vol = opt.get("volume", 0)
                oi = opt.get("openInterest", 0)
                if oi > 0 and vol > 1000 and (vol / oi) >= self.min_volume_oi_ratio:
                    unusual_strikes.append({
                        "type": "call",
                        "strike": opt["strike"],
                        "volume": vol,
                        "oi": oi
                    })

            if not unusual_strikes:
                return None

            # Calculate P/C ratio
            total_call_vol = calls["volume"].sum() if not calls.empty else 0
            total_put_vol = puts["volume"].sum() if not puts.empty else 0
            pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0

            sentiment = "bullish" if pc_ratio < 0.7 else "bearish" if pc_ratio > 1.3 else "neutral"

            return {
                "ticker": ticker,
                "source": self.name,
                "context": f"Unusual options: {len(unusual_strikes)} strikes, P/C={pc_ratio:.2f} ({sentiment})",
                "priority": "high" if sentiment == "bullish" else "medium",
                "strategy": "options_flow",
                "put_call_ratio": round(pc_ratio, 2)
            }

        except:
            return None


SCANNER_REGISTRY.register(OptionsFlowScanner)
```

---

## Phase 3: Dynamic Performance Tracking (30 min)

### Task 5: Position Tracker

**Files:**
- Create: `tradingagents/dataflows/discovery/performance/position_tracker.py`

**Implementation:**

```python
# tradingagents/dataflows/discovery/performance/position_tracker.py
"""Dynamic position tracking with time-series data."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class PositionTracker:
    """Track positions with continuous price monitoring."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.tracking_dir = self.data_dir / "recommendations" / "tracking"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

    def create_position(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Create new position to track."""
        ticker = recommendation["ticker"]
        entry_price = recommendation["entry_price"]
        rec_date = recommendation.get("recommendation_date", datetime.now().isoformat())

        return {
            "ticker": ticker,
            "recommendation_date": rec_date,
            "entry_price": entry_price,
            "pipeline": recommendation.get("pipeline", "unknown"),
            "scanner": recommendation.get("scanner", "unknown"),
            "strategy": recommendation.get("strategy_match", "unknown"),
            "confidence": recommendation.get("confidence", 5),
            "shares": recommendation.get("shares", 0),

            "price_history": [{
                "timestamp": rec_date,
                "price": entry_price,
                "return_pct": 0.0,
                "hours_held": 0,
                "days_held": 0
            }],

            "metrics": {
                "peak_return": 0.0,
                "current_return": 0.0,
                "days_held": 0,
                "status": "open"
            }
        }

    def update_position_price(self, position: Dict[str, Any], new_price: float,
                            timestamp: Optional[str] = None) -> Dict[str, Any]:
        """Update position with new price."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        entry_price = position["entry_price"]
        entry_time = datetime.fromisoformat(position["recommendation_date"])
        current_time = datetime.fromisoformat(timestamp)

        return_pct = ((new_price - entry_price) / entry_price) * 100.0
        time_diff = current_time - entry_time
        hours_held = time_diff.total_seconds() / 3600
        days_held = time_diff.days

        position["price_history"].append({
            "timestamp": timestamp,
            "price": new_price,
            "return_pct": round(return_pct, 2),
            "hours_held": round(hours_held, 1),
            "days_held": days_held
        })

        # Update metrics
        position["metrics"]["current_return"] = round(return_pct, 2)
        position["metrics"]["current_price"] = new_price
        position["metrics"]["days_held"] = days_held
        position["metrics"]["peak_return"] = max(
            position["metrics"]["peak_return"],
            return_pct
        )

        return position

    def save_position(self, position: Dict[str, Any]) -> None:
        """Save position to disk."""
        ticker = position["ticker"]
        rec_date = position["recommendation_date"].split("T")[0]
        filename = f"{ticker}_{rec_date}.json"
        filepath = self.tracking_dir / filename

        with open(filepath, "w") as f:
            json.dump(position, f, indent=2)

    def load_all_open_positions(self) -> List[Dict[str, Any]]:
        """Load all open positions."""
        positions = []
        for filepath in self.tracking_dir.glob("*.json"):
            with open(filepath, "r") as f:
                position = json.load(f)
                if position["metrics"]["status"] == "open":
                    positions.append(position)
        return positions
```

---

### Task 6: Position Updater Script

**Files:**
- Create: `scripts/update_positions.py`

**Implementation:**

```python
# scripts/update_positions.py
"""Update all open positions with current prices."""
import yfinance as yf
from datetime import datetime
from tradingagents.dataflows.discovery.performance.position_tracker import PositionTracker


def main():
    tracker = PositionTracker()
    positions = tracker.load_all_open_positions()

    if not positions:
        print("No open positions")
        return

    print(f"Updating {len(positions)} positions...")

    # Get unique tickers
    tickers = list(set(p["ticker"] for p in positions))

    # Fetch prices
    try:
        tickers_str = " ".join(tickers)
        data = yf.download(tickers_str, period="1d", progress=False)

        prices = {}
        if len(tickers) == 1:
            prices[tickers[0]] = float(data["Close"].iloc[-1])
        else:
            for ticker in tickers:
                try:
                    prices[ticker] = float(data["Close"][ticker].iloc[-1])
                except:
                    pass

        # Update each position
        for position in positions:
            ticker = position["ticker"]
            if ticker in prices:
                updated = tracker.update_position_price(position, prices[ticker])
                tracker.save_position(updated)
                print(f"  {ticker}: ${prices[ticker]:.2f} ({updated['metrics']['current_return']:+.1f}%)")

        print(f"✅ Updated {len(positions)} positions")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
```

---

## Phase 4: Streamlit Dashboard (60 min)

### Task 7: Install Dependencies & Create Entry Point

**Files:**
- Update: `requirements.txt`
- Create: `tradingagents/ui/dashboard.py`
- Create: `tradingagents/ui/utils.py`
- Create: `tradingagents/ui/pages/__init__.py`

**Add to requirements.txt:**
```
streamlit>=1.40.0
plotly>=5.18.0
```

**Dashboard entry point:**

```python
# tradingagents/ui/dashboard.py
"""Trading Discovery Dashboard."""
import streamlit as st

st.set_page_config(
    page_title="Trading Discovery",
    page_icon="🎯",
    layout="wide"
)

from tradingagents.ui.pages import home, todays_picks, portfolio, performance, settings


def main():
    st.sidebar.title("🎯 Trading Discovery")

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Today's Picks", "Portfolio", "Performance", "Settings"]
    )

    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")

    try:
        from tradingagents.ui.utils import load_quick_stats
        stats = load_quick_stats()
        st.sidebar.metric("Open Positions", stats.get("open_positions", 0))
        st.sidebar.metric("Win Rate", f"{stats.get('win_rate_7d', 0):.1f}%")
    except:
        pass

    # Render page
    if page == "Home":
        home.render()
    elif page == "Today's Picks":
        todays_picks.render()
    elif page == "Portfolio":
        portfolio.render()
    elif page == "Performance":
        performance.render()
    elif page == "Settings":
        settings.render()


if __name__ == "__main__":
    main()
```

**Utils:**

```python
# tradingagents/ui/utils.py
"""Dashboard utilities."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def load_statistics() -> Dict[str, Any]:
    """Load performance statistics."""
    stats_file = Path("data/recommendations/statistics.json")
    if not stats_file.exists():
        return {}
    with open(stats_file, "r") as f:
        return json.load(f)


def load_recommendations(date: str = None) -> List[Dict[str, Any]]:
    """Load recommendations for date."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    rec_file = Path(f"data/recommendations/{date}_recommendations.json")
    if not rec_file.exists():
        return []
    with open(rec_file, "r") as f:
        data = json.load(f)
        return data.get("rankings", [])


def load_open_positions() -> List[Dict[str, Any]]:
    """Load all open positions."""
    from tradingagents.dataflows.discovery.performance.position_tracker import PositionTracker
    tracker = PositionTracker()
    return tracker.load_all_open_positions()


def load_quick_stats() -> Dict[str, Any]:
    """Load sidebar quick stats."""
    stats = load_statistics()
    positions = load_open_positions()
    return {
        "open_positions": len(positions),
        "win_rate_7d": stats.get("overall_7d", {}).get("win_rate", 0)
    }
```

---

### Task 8: Home Page

**Files:**
- Create: `tradingagents/ui/pages/home.py`

```python
# tradingagents/ui/pages/home.py
"""Home page."""
import streamlit as st
import plotly.express as px
import pandas as pd
from tradingagents.ui.utils import load_statistics, load_open_positions


def render():
    st.title("🎯 Trading Discovery Dashboard")

    stats = load_statistics()
    if not stats:
        st.warning("No data. Run discovery first.")
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    overall_7d = stats.get("overall_7d", {})
    with col1:
        st.metric("Win Rate (7d)", f"{overall_7d.get('win_rate', 0):.1f}%")
    with col2:
        st.metric("Open Positions", len(load_open_positions()))
    with col3:
        st.metric("Avg Return (7d)", f"{overall_7d.get('avg_return', 0):+.1f}%")
    with col4:
        by_pipeline = stats.get("by_pipeline", {})
        if by_pipeline:
            best = max(by_pipeline.items(), key=lambda x: x[1].get("win_rate_7d", 0))
            st.metric("Best Pipeline", f"{best[0].title()} ({best[1].get('win_rate_7d', 0):.0f}%)")

    # Pipeline chart
    st.subheader("📊 Pipeline Performance")

    if by_pipeline:
        data = []
        for pipeline, d in by_pipeline.items():
            data.append({
                "Pipeline": pipeline.title(),
                "Win Rate": d.get("win_rate_7d", 0),
                "Avg Return": d.get("avg_return_7d", 0),
                "Count": d.get("count", 0)
            })

        df = pd.DataFrame(data)
        fig = px.scatter(df, x="Win Rate", y="Avg Return", size="Count", color="Pipeline",
                        title="Pipeline Performance")
        fig.add_hline(y=0, line_dash="dash")
        fig.add_vline(x=50, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
```

---

### Task 9: Today's Picks Page

**Files:**
- Create: `tradingagents/ui/pages/todays_picks.py`

```python
# tradingagents/ui/pages/todays_picks.py
"""Today's recommendations."""
import streamlit as st
from datetime import datetime
from tradingagents.ui.utils import load_recommendations


def render():
    st.title("📋 Today's Recommendations")

    today = datetime.now().strftime("%Y-%m-%d")
    recommendations = load_recommendations(today)

    if not recommendations:
        st.warning(f"No recommendations for {today}")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        pipelines = list(set(r.get("pipeline", "unknown") for r in recommendations))
        pipeline_filter = st.multiselect("Pipeline", pipelines, default=pipelines)
    with col2:
        min_confidence = st.slider("Min Confidence", 1, 10, 7)
    with col3:
        min_score = st.slider("Min Score", 0, 100, 70)

    # Apply filters
    filtered = [r for r in recommendations
                if r.get("pipeline", "unknown") in pipeline_filter
                and r.get("confidence", 0) >= min_confidence
                and r.get("final_score", 0) >= min_score]

    st.write(f"**{len(filtered)}** of **{len(recommendations)}** recommendations")

    # Display recommendations
    for i, rec in enumerate(filtered, 1):
        ticker = rec.get("ticker", "UNKNOWN")
        score = rec.get("final_score", 0)
        confidence = rec.get("confidence", 0)

        with st.expander(f"#{i} {ticker} - {rec.get('company_name', '')} (Score: {score}, Conf: {confidence}/10)"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Pipeline:** {rec.get('pipeline', 'unknown').title()}")
                st.write(f"**Scanner:** {rec.get('scanner', 'unknown')}")
                st.write(f"**Price:** ${rec.get('current_price', 0):.2f}")
                st.write(f"**Thesis:** {rec.get('reason', 'N/A')}")

            with col2:
                if st.button("✅ Enter Position", key=f"enter_{ticker}"):
                    st.info("Position entry modal (TODO)")
                if st.button("👀 Watch", key=f"watch_{ticker}"):
                    st.success(f"Added {ticker} to watchlist")
```

---

### Task 10: Portfolio Page

**Files:**
- Create: `tradingagents/ui/pages/portfolio.py`

```python
# tradingagents/ui/pages/portfolio.py
"""Portfolio tracker."""
import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime
from tradingagents.ui.utils import load_open_positions


def render():
    st.title("💼 Portfolio Tracker")

    # Manual add form
    with st.expander("➕ Add Position"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ticker = st.text_input("Ticker")
        with col2:
            entry_price = st.number_input("Entry Price", min_value=0.0)
        with col3:
            shares = st.number_input("Shares", min_value=0, step=1)
        with col4:
            st.write("")  # Spacing
            if st.button("Add"):
                if ticker and entry_price > 0 and shares > 0:
                    from tradingagents.dataflows.discovery.performance.position_tracker import PositionTracker
                    tracker = PositionTracker()
                    pos = tracker.create_position({
                        "ticker": ticker.upper(),
                        "entry_price": entry_price,
                        "shares": shares,
                        "recommendation_date": datetime.now().isoformat(),
                        "pipeline": "manual",
                        "scanner": "manual",
                        "strategy_match": "manual",
                        "confidence": 5
                    })
                    tracker.save_position(pos)
                    st.success(f"Added {ticker.upper()}")
                    st.rerun()

    # Load positions
    positions = load_open_positions()

    if not positions:
        st.info("No open positions")
        return

    # Summary
    total_invested = sum(p["entry_price"] * p.get("shares", 0) for p in positions)
    total_current = sum(p["metrics"]["current_price"] * p.get("shares", 0) for p in positions)
    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Invested", f"${total_invested:,.0f}")
    with col2:
        st.metric("Current", f"${total_current:,.0f}")
    with col3:
        st.metric("P/L", f"${total_pnl:,.0f}", delta=f"{total_pnl_pct:+.1f}%")
    with col4:
        st.metric("Positions", len(positions))

    # Table
    st.subheader("📊 Positions")

    data = []
    for p in positions:
        pnl = (p["metrics"]["current_price"] - p["entry_price"]) * p.get("shares", 0)
        data.append({
            "Ticker": p["ticker"],
            "Entry": f"${p['entry_price']:.2f}",
            "Current": f"${p['metrics']['current_price']:.2f}",
            "Shares": p.get("shares", 0),
            "P/L": f"${pnl:.2f}",
            "P/L %": f"{p['metrics']['current_return']:+.1f}%",
            "Days": p["metrics"]["days_held"]
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
```

---

### Task 11: Performance & Settings Pages (Simplified)

**Files:**
- Create: `tradingagents/ui/pages/performance.py`
- Create: `tradingagents/ui/pages/settings.py`

```python
# tradingagents/ui/pages/performance.py
"""Performance analytics."""
import streamlit as st
import plotly.express as px
import pandas as pd
from tradingagents.ui.utils import load_statistics


def render():
    st.title("📊 Performance Analytics")

    stats = load_statistics()
    if not stats:
        st.warning("No data available")
        return

    # Scanner heatmap
    st.subheader("🔥 Scanner Performance")

    by_scanner = stats.get("by_scanner", {})
    if by_scanner:
        data = []
        for scanner, d in by_scanner.items():
            data.append({
                "Scanner": scanner,
                "Win Rate": d.get("win_rate_7d", 0),
                "Avg Return": d.get("avg_return_7d", 0),
                "Count": d.get("count", 0)
            })

        df = pd.DataFrame(data)
        fig = px.scatter(df, x="Win Rate", y="Avg Return", size="Count",
                        color="Win Rate", hover_data=["Scanner"],
                        color_continuous_scale="RdYlGn")
        fig.add_hline(y=0, line_dash="dash")
        fig.add_vline(x=50, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
```

```python
# tradingagents/ui/pages/settings.py
"""Settings page."""
import streamlit as st
from tradingagents.default_config import DEFAULT_CONFIG


def render():
    st.title("⚙️ Settings")

    st.info("Configuration UI - TODO: Implement save functionality")

    # Show current config
    config = DEFAULT_CONFIG.get("discovery", {})

    st.subheader("Pipelines")
    pipelines = config.get("pipelines", {})
    for name, cfg in pipelines.items():
        with st.expander(f"{name.title()} Pipeline"):
            st.write(f"Enabled: {cfg.get('enabled')}")
            st.write(f"Priority: {cfg.get('priority')}")
            st.write(f"Budget: {cfg.get('deep_dive_budget')}")

    st.subheader("Scanners")
    scanners = config.get("scanners", {})
    for name, cfg in scanners.items():
        st.checkbox(f"{name}", value=cfg.get("enabled"), key=f"scan_{name}")
```

**Create __init__.py:**

```python
# tradingagents/ui/pages/__init__.py
from . import home, todays_picks, portfolio, performance, settings
```

---

## Phase 5: Integration (15 min)

### Task 12: Update Discovery Graph

**Files:**
- Modify: `tradingagents/graph/discovery_graph.py`

**Add to imports:**
```python
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
```

**Replace scanner_node() method:**

```python
def scanner_node(self, state: DiscoveryState) -> Dict[str, Any]:
    """Scan using modular registry."""
    print("🔍 Scanning market for opportunities...")

    # Performance tracking
    try:
        self.analytics.update_performance_tracking()
    except Exception as e:
        print(f"   Warning: {e}")

    state.setdefault("tool_logs", [])

    # Collect by pipeline
    pipeline_candidates = {
        "edge": [], "momentum": [], "news": [], "social": [], "events": []
    }

    pipeline_config = self.config.get("discovery", {}).get("pipelines", {})

    # Run enabled scanners
    for scanner_class in SCANNER_REGISTRY.get_all_scanners():
        pipeline = scanner_class.pipeline

        if not pipeline_config.get(pipeline, {}).get("enabled", True):
            continue

        try:
            scanner = scanner_class(self.config)
            if not scanner.is_enabled():
                continue

            state["tool_executor"] = self._execute_tool_logged
            candidates = scanner.scan(state)
            pipeline_candidates[pipeline].extend(candidates)

        except Exception as e:
            print(f"   Error in {scanner_class.name}: {e}")

    # Merge candidates
    all_candidates = []
    for candidates in pipeline_candidates.values():
        all_candidates.extend(candidates)

    unique_candidates = {}
    self._merge_candidates_into_dict(all_candidates, unique_candidates)

    final = list(unique_candidates.values())
    print(f"   Found {len(final)} unique candidates")

    return {
        "tickers": [c["ticker"] for c in final],
        "candidate_metadata": final,
        "tool_logs": state.get("tool_logs", []),
        "status": "scanned"
    }
```

---

## Summary & Running

**What's Implemented:**
1. ✅ Modular scanner registry
2. ✅ Config with pipelines/scanners
3. ✅ 2 edge scanners (insider, options) as templates
4. ✅ Dynamic position tracker
5. ✅ Position updater script
6. ✅ Full Streamlit dashboard (5 pages)
7. ✅ Discovery graph integration

**To Run:**

```bash
# Install dependencies
pip install streamlit plotly

# Update positions (run hourly)
python scripts/update_positions.py

# Start dashboard
streamlit run tradingagents/ui/dashboard.py

# Run discovery
python -m cli.main analyze  # Select discovery mode
```

**Next Steps:**
1. Test discovery with new architecture
2. Add more edge scanners (congress, 13F)
3. Add tests/docs when ready
4. Tune scanner limits based on performance

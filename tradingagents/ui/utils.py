"""
Utility functions for the Trading Agents Dashboard.

This module provides helper functions for loading data from various sources
including statistics, recommendations, positions, and quick metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


def get_data_directory() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / "data"


def load_statistics() -> Dict[str, Any]:
    """
    Load statistics data from JSON file.

    Returns:
        Dictionary containing statistics data
    """
    stats_file = get_data_directory() / "recommendations" / "statistics.json"

    if not stats_file.exists():
        return {}

    try:
        with open(stats_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading statistics: {e}")
        return {}


def _extract_date_from_filename(filename: str) -> Optional[str]:
    name = filename
    if name.endswith("_recommendations.json"):
        date_str = name[: -len("_recommendations.json")]
    elif name.endswith(".json"):
        date_str = name[:-5]
    else:
        return None

    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        return None


def _find_latest_recommendations_file(
    recommendations_dir: Path,
) -> Tuple[Optional[Path], Optional[str]]:
    if not recommendations_dir.exists():
        return None, None

    ignore = {"statistics.json", "performance_database.json"}
    dated_files: List[Tuple[str, Path]] = []
    for path in recommendations_dir.glob("*.json"):
        if path.name in ignore:
            continue
        date_str = _extract_date_from_filename(path.name)
        if date_str:
            dated_files.append((date_str, path))

    if not dated_files:
        return None, None

    dated_files.sort(key=lambda item: item[0])
    latest_date, latest_path = dated_files[-1]
    return latest_path, latest_date


def _load_recommendations_payload(
    rec_file: Path,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        with open(rec_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading recommendations from {rec_file}: {e}")
        return [], None

    if isinstance(data, dict):
        return data.get("recommendations", []) or [], data.get("date")
    if isinstance(data, list):
        return data, None
    return [], None


def load_recommendations(
    date: Optional[str] = None, *, return_meta: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Load recommendations data from JSON file.

    Args:
        date: Optional date in format YYYY-MM-DD. If None, loads today's recommendations.
        return_meta: If True, returns (recommendations, meta) tuple.

    Returns:
        List of recommendation dictionaries
    """
    requested_date = date or datetime.now().strftime("%Y-%m-%d")
    recommendations_dir = get_data_directory() / "recommendations"

    candidates = [
        recommendations_dir / f"{requested_date}_recommendations.json",
        recommendations_dir / f"{requested_date}.json",
    ]

    rec_file = next((p for p in candidates if p.exists()), None)
    used_date = requested_date
    is_fallback = False

    if rec_file is None and date is None:
        rec_file, latest_date = _find_latest_recommendations_file(recommendations_dir)
        if rec_file is not None:
            used_date = latest_date or requested_date
            is_fallback = True

    if rec_file is None:
        meta = {
            "requested_date": requested_date,
            "date": None,
            "source_file": None,
            "is_fallback": False,
        }
        return ([], meta) if return_meta else []

    recommendations, payload_date = _load_recommendations_payload(rec_file)
    if payload_date:
        used_date = payload_date

    meta = {
        "requested_date": requested_date,
        "date": used_date,
        "source_file": str(rec_file),
        "is_fallback": is_fallback,
    }
    return (recommendations, meta) if return_meta else recommendations


def load_open_positions() -> List[Dict[str, Any]]:
    """
    Load open positions from the position tracker.

    Returns:
        List of open position dictionaries
    """
    try:
        from tradingagents.dataflows.discovery.performance.position_tracker import PositionTracker

        tracker = PositionTracker()
        positions = tracker.load_all_open_positions()
        return positions if positions else []
    except Exception as e:
        logger.error(f"Error loading open positions: {e}")
        return []


def load_performance_database() -> List[Dict[str, Any]]:
    """
    Load the performance database (flattened list of recommendations).
    """
    db_file = get_data_directory() / "recommendations" / "performance_database.json"
    if not db_file.exists():
        return []

    try:
        with open(db_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading performance database: {e}")
        return []

    if isinstance(data, dict):
        by_date = data.get("recommendations_by_date", {})
        recs: List[Dict[str, Any]] = []
        for items in by_date.values():
            if isinstance(items, list):
                recs.extend(items)
        return recs

    if isinstance(data, list):
        return data

    return []


def load_strategy_metrics() -> List[Dict[str, Any]]:
    """
    Build per-strategy metrics from the performance database if available.
    Falls back to statistics.json when performance database is missing.
    """
    recs = load_performance_database()
    if recs:
        metrics: Dict[str, Dict[str, float]] = {}
        for rec in recs:
            strategy = rec.get("strategy_match", "unknown")
            if strategy not in metrics:
                metrics[strategy] = {
                    "count": 0,
                    "wins": 0,
                    "sum_return": 0.0,
                }

            if "return_7d" in rec:
                metrics[strategy]["count"] += 1
                metrics[strategy]["sum_return"] += float(rec.get("return_7d", 0.0) or 0.0)
                if rec.get("win_7d"):
                    metrics[strategy]["wins"] += 1

        results = []
        for strategy, data in metrics.items():
            count = int(data["count"])
            if count == 0:
                continue
            win_rate = round((data["wins"] / count) * 100, 1)
            avg_return = round(data["sum_return"] / count, 2)
            results.append(
                {
                    "Strategy": strategy,
                    "Win Rate": win_rate,
                    "Avg Return": avg_return,
                    "Count": count,
                }
            )
        return results

    stats = load_statistics()
    by_strategy = stats.get("by_strategy", {}) if stats else {}
    results = []
    for strategy, data in by_strategy.items():
        win_rate = data.get("win_rate_7d") or data.get("win_rate", 0)
        avg_return = data.get("avg_return_7d", 0)
        count = data.get("wins_7d", 0) + data.get("losses_7d", 0)
        results.append(
            {
                "Strategy": strategy,
                "Win Rate": win_rate,
                "Avg Return": avg_return,
                "Count": count,
            }
        )
    return results


def load_quick_stats() -> Tuple[int, float]:
    """
    Load quick statistics for the sidebar.

    Returns:
        Tuple of (open_positions_count, win_rate_percentage)
    """
    # Load open positions
    positions = load_open_positions()
    open_positions_count = len(positions)

    # Calculate win rate from statistics
    stats = load_statistics()
    win_rate = 0.0

    if stats and "trades" in stats and len(stats["trades"]) > 0:
        winning_trades = sum(
            1
            for trade in stats["trades"]
            if trade.get("status") == "closed" and trade.get("profit", 0) > 0
        )
        total_trades = sum(1 for trade in stats["trades"] if trade.get("status") == "closed")
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100

    return open_positions_count, win_rate

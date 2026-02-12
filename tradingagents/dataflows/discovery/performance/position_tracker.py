"""
Position Tracker Module

Monitors positions continuously with dynamic price history tracking.
Maintains complete price time-series and calculates real-time metrics.
"""

import json
from datetime import datetime
from pathlib import Path

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)
from typing import Any, Dict, List, Optional


class PositionTracker:
    """
    Dynamic position tracking system that monitors positions continuously.
    Maintains complete price history and calculates real-time metrics.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize PositionTracker.

        Args:
            data_dir: Root directory for position storage (default: "data")
        """
        self.data_dir = Path(data_dir)
        self.positions_dir = self.data_dir / "positions"
        self.positions_dir.mkdir(parents=True, exist_ok=True)

    def create_position(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new position dictionary from a recommendation.

        Args:
            recommendation: Recommendation dict with at minimum:
                - ticker: Stock ticker
                - entry_price: Entry price for the position
                - recommendation_date: Date of recommendation
                - scanner: Source scanner
                - strategy: Strategy name
                - pipeline: Pipeline identifier
                - confidence: Confidence score (0-1)
                - shares: Number of shares to buy

        Returns:
            Position dictionary with initialized structure
        """
        now = datetime.utcnow()
        position = {
            "ticker": recommendation.get("ticker"),
            "entry_price": recommendation.get("entry_price"),
            "recommendation_date": recommendation.get("recommendation_date"),
            "pipeline": recommendation.get("pipeline"),
            "scanner": recommendation.get("scanner"),
            "strategy": recommendation.get("strategy"),
            "confidence": recommendation.get("confidence"),
            "shares": recommendation.get("shares"),
            "created_at": now.isoformat(),
            "status": "open",
            "price_history": [
                {
                    "timestamp": now.isoformat(),
                    "price": recommendation.get("entry_price"),
                    "return_pct": 0.0,
                    "hours_held": 0.0,
                    "days_held": 0.0,
                }
            ],
            "metrics": {
                "peak_return": 0.0,
                "current_return": 0.0,
                "current_price": recommendation.get("entry_price"),
                "days_held": 0.0,
                "status": "open",
            },
        }
        return position

    def update_position_price(
        self,
        position: Dict[str, Any],
        new_price: float,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update position with new price point and recalculate metrics.

        Args:
            position: Position dictionary to update
            new_price: New price to add to history
            timestamp: ISO timestamp for price (default: current UTC time)

        Returns:
            Updated position dictionary
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            price_time = datetime.fromisoformat(timestamp)
        else:
            price_time = timestamp

        # Get entry time from recommendation_date or created_at
        if isinstance(position["recommendation_date"], str):
            entry_time = datetime.fromisoformat(position["recommendation_date"])
        else:
            entry_time = datetime.fromisoformat(position["created_at"])

        # Calculate time differences
        time_diff = price_time - entry_time
        hours_held = time_diff.total_seconds() / 3600
        days_held = time_diff.total_seconds() / (3600 * 24)

        # Calculate returns
        entry_price = position["entry_price"]
        return_pct = ((new_price - entry_price) / entry_price) * 100

        # Create price history entry
        price_entry = {
            "timestamp": timestamp,
            "price": new_price,
            "return_pct": return_pct,
            "hours_held": hours_held,
            "days_held": days_held,
        }

        # Add to price history
        position["price_history"].append(price_entry)

        # Update metrics
        position["metrics"]["current_price"] = new_price
        position["metrics"]["current_return"] = return_pct
        position["metrics"]["days_held"] = days_held

        # Update peak return if current return is higher
        if return_pct > position["metrics"]["peak_return"]:
            position["metrics"]["peak_return"] = return_pct

        return position

    def save_position(self, position: Dict[str, Any]) -> str:
        """
        Save position to JSON file.

        Creates file: {ticker}_{created_at_timestamp}.json

        Args:
            position: Position dictionary to save

        Returns:
            Path to saved file
        """
        ticker = position["ticker"]
        created_at = position["created_at"]

        # Parse created_at to create a filename-safe timestamp
        created_dt = datetime.fromisoformat(created_at)
        timestamp_str = created_dt.strftime("%Y%m%d_%H%M%S")

        filename = f"{ticker}_{timestamp_str}.json"
        filepath = self.positions_dir / filename

        with open(filepath, "w") as f:
            json.dump(position, f, indent=2)

        return str(filepath)

    def load_all_open_positions(self) -> List[Dict[str, Any]]:
        """
        Load all positions with status="open" from disk.

        Returns:
            List of position dictionaries
        """
        open_positions = []

        if not self.positions_dir.exists():
            return open_positions

        for filepath in self.positions_dir.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    position = json.load(f)
                    if position.get("status") == "open":
                        open_positions.append(position)
            except (json.JSONDecodeError, IOError) as e:
                # Log error but continue loading other positions
                logger.error(f"Error loading position from {filepath}: {e}")

        return open_positions

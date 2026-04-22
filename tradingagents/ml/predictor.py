"""ML predictor for discovery pipeline — loads trained model and runs inference.

Gracefully degrades: if no model file exists, all predictions return None.
The discovery pipeline works exactly as before without a trained model.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tradingagents.ml.feature_engineering import FEATURE_COLUMNS
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

# Default model path relative to project root
DEFAULT_MODEL_DIR = Path("data/ml")
MODEL_FILENAME = "tabpfn_model.pkl"
METRICS_FILENAME = "metrics.json"

# Class label mapping
LABEL_MAP = {-1: "LOSS", 0: "TIMEOUT", 1: "WIN"}


class LGBMWrapper:
    """Sklearn-compatible wrapper for LightGBM booster with original label mapping.

    Defined here (not in train script) so pickle can find the class on deserialization.
    """

    def __init__(self, booster, y_train=None):
        self.booster = booster
        self.classes_ = np.array([-1, 0, 1])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.booster.predict(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        mapped = np.argmax(probas, axis=1)
        return self.classes_[mapped]


class MLPredictor:
    """Wraps a trained ML model for win probability prediction.

    Usage:
        predictor = MLPredictor.load()  # loads from default path
        predictor.load_market_context()  # pre-fetch SPY/VIX once per scan run
        if predictor is not None:
            result = predictor.predict(feature_dict)
            # result = {"win_prob": 0.73, "loss_prob": 0.27, "prediction": "WIN"}
    """

    def __init__(self, model: Any, feature_columns: List[str], model_type: str = "tabpfn"):
        self.model = model
        self.feature_columns = feature_columns
        self.model_type = model_type
        self._market_ctx: Optional[pd.DataFrame] = None

    @classmethod
    def load(cls, model_dir: Optional[str] = None) -> Optional[MLPredictor]:
        """Load a trained model from disk. Returns None if no model exists."""
        import json

        if model_dir is None:
            model_dir = str(DEFAULT_MODEL_DIR)

        model_path = os.path.join(model_dir, MODEL_FILENAME)
        if not os.path.exists(model_path):
            logger.debug(f"No ML model found at {model_path} — ML predictions disabled")
            return None

        try:
            with open(model_path, "rb") as f:
                saved = pickle.load(f)

            model = saved["model"]
            feature_columns = saved.get("feature_columns", FEATURE_COLUMNS)
            model_type = saved.get("model_type", "unknown")

            # Load metrics and warn if model is stale
            metrics_path = os.path.join(model_dir, METRICS_FILENAME)
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                training_date_str = metrics.get("training_date")
                if training_date_str:
                    age_days = (pd.Timestamp.today() - pd.Timestamp(training_date_str)).days
                    if age_days > 30:
                        logger.warning(
                            f"ML model is {age_days} days old (trained {training_date_str}) — "
                            f"consider retraining: python scripts/train_ml_model.py"
                        )
                else:
                    logger.warning(
                        "ML model has no training_date in metrics.json — staleness cannot be assessed"
                    )

            logger.info(f"Loaded ML model ({model_type}) from {model_path}")
            return cls(model=model, feature_columns=feature_columns, model_type=model_type)

        except Exception as e:
            logger.warning(f"Failed to load ML model from {model_path}: {e}")
            return None

    def load_market_context(self, lookback_days: int = 60) -> None:
        """Pre-fetch SPY/VIX context for recent N days. Call once before a batch of predict() calls."""
        from tradingagents.ml.feature_engineering import fetch_market_context

        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
        self._market_ctx = fetch_market_context(start, end)
        logger.info(f"Loaded market context: {len(self._market_ctx)} trading days")

    def predict(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Predict win probability for a single candidate.

        Args:
            features: Dict mapping feature names to values (from feature_engineering).
                      If load_market_context() was called, regime features are injected
                      automatically from the most recent available market data.

        Returns:
            Dict with win_prob, loss_prob, prediction, or None on error.
        """
        try:
            # Inject market regime features from pre-fetched context
            if self._market_ctx is not None and not self._market_ctx.empty:
                latest = self._market_ctx.iloc[-1]
                features = dict(features)  # don't mutate caller's dict
                features["spy_return_20d"] = float(latest.get("spy_return_20d", np.nan))
                features["vix_level"] = float(latest.get("vix_level", np.nan))
                features["vix_ma20_ratio"] = float(latest.get("vix_ma20_ratio", np.nan))
                features["stock_vs_spy_20d"] = (
                    features.get("return_20d", np.nan) - features["spy_return_20d"]
                )
                # sector_return_20d left as NaN unless separately injected by scanner

            # Build feature vector in correct order
            X = np.array([[features.get(col, np.nan) for col in self.feature_columns]])
            X_df = pd.DataFrame(X, columns=self.feature_columns)

            # Get probability predictions
            probas = self.model.predict_proba(X_df)
            proba_row = probas[0]

            # Handle binary model (classes [0, 1]) and legacy 3-class model
            classes = list(self.model.classes_)
            if len(classes) == 2:
                # Binary: [P(NOT-WIN), P(WIN)]
                win_prob = float(proba_row[1])
                loss_prob = float(proba_row[0])
                prediction = "WIN" if win_prob >= 0.5 else "NOT-WIN"
            else:
                # Legacy 3-class: classes [-1, 0, 1] or [0, 1, 2]
                win_idx = classes.index(1) if 1 in classes else classes.index(2)
                loss_idx = classes.index(-1) if -1 in classes else classes.index(0)
                win_prob = float(proba_row[win_idx])
                loss_prob = float(proba_row[loss_idx])
                pred_class = classes[int(np.argmax(proba_row))]
                prediction = LABEL_MAP.get(pred_class, str(pred_class))

            return {
                "win_prob": win_prob,
                "loss_prob": loss_prob,
                "prediction": prediction,
            }

        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None

    def predict_batch(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[Optional[Dict[str, Any]]]:
        """Predict win probabilities for multiple candidates."""
        return [self.predict(f) for f in feature_dicts]

    def save(self, model_dir: Optional[str] = None) -> str:
        """Save the model to disk."""
        if model_dir is None:
            model_dir = str(DEFAULT_MODEL_DIR)

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, MODEL_FILENAME)

        saved = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type,
        }

        with open(model_path, "wb") as f:
            pickle.dump(saved, f)

        logger.info(f"Saved ML model to {model_path}")
        return model_path

#!/usr/bin/env python3
"""Train ML model on the generated dataset.

Supports TabPFN (recommended, requires GPU or API) and LightGBM (fallback).
Uses time-based train/validation split to prevent data leakage.

Usage:
    python scripts/train_ml_model.py
    python scripts/train_ml_model.py --model lightgbm
    python scripts/train_ml_model.py --model tabpfn --dataset data/ml/training_dataset.parquet
    python scripts/train_ml_model.py --max-train-samples 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tradingagents.ml.feature_engineering import FEATURE_COLUMNS
from tradingagents.ml.predictor import MLPredictor
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path("data/ml")
LABEL_NAMES = {0: "NOT-WIN", 1: "WIN"}


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the training dataset."""
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} samples from {path}")

    # Validate columns
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column")
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column")

    # Show label distribution
    unique_labels = sorted(df["label"].dropna().unique())
    for label in unique_labels:
        name = LABEL_NAMES.get(int(label), str(int(label)))
        count = (df["label"] == label).sum()
        pct = count / len(df) * 100
        logger.info(f"  {name:>8} ({int(label):+d}): {count:>7} ({pct:.1f}%)")

    return df


def time_split(
    df: pd.DataFrame,
    val_start: str = "2024-07-01",
    max_train_samples: int | None = None,
) -> tuple:
    """Split dataset by time — train on older data, validate on newer."""
    df["date"] = pd.to_datetime(df["date"])
    val_start_dt = pd.Timestamp(val_start)

    train = df[df["date"] < val_start_dt].copy()
    val = df[df["date"] >= val_start_dt].copy()

    if max_train_samples is not None and len(train) > max_train_samples:
        train = train.sort_values("date").tail(max_train_samples)
        logger.info(
            f"Limiting training samples to most recent {max_train_samples} " f"before {val_start}"
        )

    logger.info(f"Time-based split at {val_start}:")
    logger.info(
        f"  Train: {len(train)} samples ({train['date'].min().date()} to {train['date'].max().date()})"
    )
    logger.info(
        f"  Val:   {len(val)} samples ({val['date'].min().date()} to {val['date'].max().date()})"
    )

    # Return DataFrames (not numpy) so LGBMClassifier preserves feature names
    X_train = train[FEATURE_COLUMNS]
    y_train = train["label"].astype(int)
    X_val = val[FEATURE_COLUMNS]
    y_val = val["label"].astype(int)

    return X_train, y_train, X_val, y_val


def train_tabpfn(X_train, y_train, X_val, y_val):
    """Train using TabPFN foundation model."""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        logger.error("TabPFN not installed. Install with: pip install tabpfn")
        logger.error("Falling back to LightGBM...")
        return train_lightgbm(X_train, y_train, X_val, y_val)

    logger.info("Training TabPFN classifier...")

    # TabPFN handles NaN values natively
    # For large datasets, subsample training data (TabPFN works best with <10K samples)
    max_train = 10_000
    if len(X_train) > max_train:
        logger.info(f"Subsampling training data: {len(X_train)} → {max_train}")
        idx = np.random.RandomState(42).choice(len(X_train), max_train, replace=False)
        X_train_sub = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
        y_train_sub = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    try:
        clf = TabPFNClassifier()
        clf.fit(X_train_sub, y_train_sub)
        return clf, "tabpfn"
    except Exception as e:
        logger.error(f"TabPFN training failed: {e}")
        logger.error("Falling back to LightGBM...")
        return train_lightgbm(X_train, y_train, X_val, y_val)


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train binary LightGBM + apply isotonic calibration on held-out val set."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Install with: pip install lightgbm")
        sys.exit(1)

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator  # sklearn ≥ 1.2

    logger.info("Training binary LightGBM classifier...")

    booster = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=63,
        max_depth=8,
        min_child_samples=100,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_gain_to_split=0.01,
        verbose=-1,
        random_state=42,
        n_jobs=-1,
    )
    booster.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=100),
        ],
    )

    logger.info(f"Best iteration: {booster.best_iteration_}")

    # Isotonic calibration (≥1000 val samples) or Platt scaling (<1000)
    method = "isotonic" if len(X_val) >= 1000 else "sigmoid"
    logger.info(f"Applying {method} calibration on {len(X_val)} val samples...")
    calibrated = CalibratedClassifierCV(FrozenEstimator(booster), method=method)
    calibrated.fit(X_val, y_val)

    return calibrated, "lightgbm_binary_calibrated"


def evaluate(model, X_val, y_val, model_type: str) -> dict:
    """Evaluate binary model and return metrics dict."""
    from sklearn.metrics import roc_auc_score

    if isinstance(X_val, np.ndarray):
        X_df = pd.DataFrame(X_val, columns=FEATURE_COLUMNS)
    else:
        X_df = X_val

    y_val_arr = np.array(y_val)
    y_pred = model.predict(X_df)
    probas = model.predict_proba(X_df)

    # Binary model: col 1 = P(WIN)
    win_col_idx = 1
    win_probs_all = probas[:, win_col_idx]

    accuracy = accuracy_score(y_val_arr, y_pred)
    report = classification_report(
        y_val_arr,
        y_pred,
        target_names=["NOT-WIN (0)", "WIN (+1)"],
        output_dict=True,
    )
    cm = confusion_matrix(y_val_arr, y_pred)
    roc_auc = roc_auc_score(y_val_arr, win_probs_all)

    # Avg P(WIN) for actual winners
    win_mask = y_val_arr == 1
    avg_win_prob_for_actual_wins = (
        float(win_probs_all[win_mask].mean()) if win_mask.sum() > 0 else 0.0
    )

    # High-confidence win precision (threshold 0.55 — above coin flip after calibration)
    high_conf_mask = win_probs_all >= 0.55
    high_conf_precision = (
        float((y_val_arr[high_conf_mask] == 1).mean()) if high_conf_mask.sum() > 0 else 0.0
    )
    high_conf_count = int(high_conf_mask.sum())

    # Quintile calibration
    quintile_labels = pd.qcut(win_probs_all, q=5, labels=False, duplicates="drop")
    calibration = {}
    for q in sorted(set(quintile_labels)):
        mask = quintile_labels == q
        q_probs = win_probs_all[mask]
        calibration[f"Q{q+1}"] = {
            "mean_predicted_win_prob": round(float(q_probs.mean()), 4),
            "actual_win_rate": round(float((y_val_arr[mask] == 1).mean()), 4),
            "count": int(mask.sum()),
        }

    # Top decile
    top_decile_threshold = np.percentile(win_probs_all, 90)
    top_decile_mask = win_probs_all >= top_decile_threshold
    top_decile_win_rate = (
        float((y_val_arr[top_decile_mask] == 1).mean()) if top_decile_mask.sum() > 0 else 0.0
    )

    metrics = {
        "model_type": model_type,
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "win_precision": round(report["WIN (+1)"]["precision"], 4),
        "win_recall": round(report["WIN (+1)"]["recall"], 4),
        "win_f1": round(report["WIN (+1)"]["f1-score"], 4),
        "win_class_prevalence": round(float(win_mask.mean()), 4),
        "confusion_matrix": cm.tolist(),
        "avg_win_prob_for_actual_wins": round(avg_win_prob_for_actual_wins, 4),
        "high_confidence_win_precision": round(high_conf_precision, 4),
        "high_confidence_win_count": high_conf_count,
        "calibration_quintiles": calibration,
        "top_decile_win_rate": round(top_decile_win_rate, 4),
        "top_decile_threshold": round(float(top_decile_threshold), 4),
        "top_decile_count": int(top_decile_mask.sum()),
        "val_samples": len(y_val_arr),
        "training_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
    }

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {model_type}")
    logger.info(f"Accuracy:      {accuracy:.1%}")
    logger.info(f"ROC-AUC:       {roc_auc:.3f}")
    logger.info(
        f"WIN Precision: {metrics['win_precision']:.3f}  Recall: {metrics['win_recall']:.3f}  F1: {metrics['win_f1']:.3f}"
    )
    logger.info(f"WIN prevalence in val: {float(win_mask.mean()):.1%}")
    logger.info(f"\nAvg P(WIN) for actual winners: {avg_win_prob_for_actual_wins:.1%}")
    logger.info(
        f"High-confidence (≥55%) precision: {high_conf_precision:.1%} ({high_conf_count} samples)"
    )

    logger.info("\nCalibration (does higher P(WIN) = more actual wins?):")
    logger.info(f"{'Quintile':>10} {'Avg P(WIN)':>12} {'Actual WIN%':>12} {'Count':>8}")
    for q_name, q_data in calibration.items():
        logger.info(
            f"{q_name:>10} {q_data['mean_predicted_win_prob']:>12.1%} "
            f"{q_data['actual_win_rate']:>12.1%} {q_data['count']:>8}"
        )

    baseline_win = float(win_mask.mean())
    logger.info(
        f"\nTop decile: P(WIN) >= {top_decile_threshold:.1%}, actual WR = {top_decile_win_rate:.1%} "
        f"({int(top_decile_mask.sum())} samples, {top_decile_win_rate/baseline_win:.2f}x lift)"
    )
    logger.info(f"{'='*60}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ML model for win probability")
    parser.add_argument("--dataset", type=str, default="data/ml/training_dataset.parquet")
    parser.add_argument(
        "--model",
        type=str,
        choices=["tabpfn", "lightgbm", "auto"],
        default="auto",
        help="Model type (auto tries TabPFN first, falls back to LightGBM)",
    )
    parser.add_argument(
        "--val-start",
        type=str,
        default="2024-07-01",
        help="Validation split date (default: 2024-07-01)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training samples to the most recent N before val-start",
    )
    parser.add_argument("--output-dir", type=str, default="data/ml")
    args = parser.parse_args()

    if args.max_train_samples is not None and args.max_train_samples <= 0:
        logger.error("--max-train-samples must be a positive integer")
        sys.exit(1)

    # Load dataset
    df = load_dataset(args.dataset)

    # Split
    X_train, y_train, X_val, y_val = time_split(
        df,
        val_start=args.val_start,
        max_train_samples=args.max_train_samples,
    )

    if len(X_val) == 0:
        logger.error(f"No validation data after {args.val_start} — adjust --val-start")
        sys.exit(1)

    # Train
    if args.model == "tabpfn" or args.model == "auto":
        model, model_type = train_tabpfn(X_train, y_train, X_val, y_val)
    else:
        model, model_type = train_lightgbm(X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = evaluate(model, X_val, y_val, model_type)

    # Save model
    predictor = MLPredictor(model=model, feature_columns=FEATURE_COLUMNS, model_type=model_type)
    model_path = predictor.save(args.output_dir)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Backtest picks feature analysis — find what separates winners from losers.

Loads picks.csv from a backtest run and performs correlation analysis between
the numeric fields each scanner emits (e.g. vol_ratio, pct_from_52w_high,
atr_ratio) and 20d forward returns. Reveals which sub-conditions within a
signal predict success — enabling threshold tuning without a new research round.

Usage:
    python scripts/analyze_backtest_picks.py
    python scripts/analyze_backtest_picks.py --path results/backtest/2026-04-16/picks.csv
    python scripts/analyze_backtest_picks.py --scanner volume_dry_up
    python scripts/analyze_backtest_picks.py --horizon 10

Output:
    - Feature correlation table (per scanner)
    - Win-rate by percentile bucket for key numeric features
    - Suggested threshold adjustments
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

WINDOWS = [1, 5, 10, 20]


def load_picks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def find_latest_picks(base_dir: str = "results/backtest") -> Path:
    runs = sorted(Path(base_dir).glob("*/picks.csv"))
    if not runs:
        raise FileNotFoundError(f"No picks.csv found under {base_dir}")
    return runs[-1]


def numeric_features(df: pd.DataFrame) -> list[str]:
    """Return scanner-emitted numeric columns (not forward returns, not metadata)."""
    exclude = {
        "date", "scanner", "ticker", "priority", "context", "source", "strategy",
        "fwd_1d", "fwd_5d", "fwd_10d", "fwd_20d",
    }
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def correlation_table(sub: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Pearson + Spearman correlation of each numeric feature vs fwd return."""
    fwd_col = f"fwd_{horizon}d"
    valid = sub.dropna(subset=[fwd_col])
    if valid.empty:
        return pd.DataFrame()

    features = [f for f in numeric_features(valid) if valid[f].notna().sum() > 10]
    rows = []
    for feat in features:
        col = valid[feat].dropna()
        shared = valid.loc[col.index, fwd_col].dropna()
        col = col.loc[shared.index]
        if len(col) < 10:
            continue
        pearson = col.corr(shared)
        spearman = col.rank().corr(shared.rank())
        rows.append({
            "feature": feat,
            "n": len(col),
            "pearson_r": round(pearson, 3),
            "spearman_r": round(spearman, 3),
            "abs_spearman": abs(spearman),
        })

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)
    return result.drop(columns=["abs_spearman"])


def bucket_analysis(sub: pd.DataFrame, feature: str, horizon: int, n_buckets: int = 4) -> pd.DataFrame:
    """Win rate and avg return by quartile of a feature."""
    fwd_col = f"fwd_{horizon}d"
    valid = sub[[feature, fwd_col]].dropna()
    if len(valid) < n_buckets * 3:
        return pd.DataFrame()

    labels = [f"Q{i+1}" for i in range(n_buckets)]
    try:
        valid = valid.copy()
        valid["bucket"] = pd.qcut(valid[feature], q=n_buckets, labels=labels, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rows = []
    for label, grp in valid.groupby("bucket", observed=True):
        rets = grp[fwd_col].dropna()
        rows.append({
            "bucket": label,
            f"{feature}_range": f"{grp[feature].min():.2g}–{grp[feature].max():.2g}",
            "n": len(rets),
            f"win_rate_{horizon}d": round((rets > 0).mean() * 100, 1) if len(rets) else None,
            f"avg_return_{horizon}d": round(rets.mean() * 100, 2) if len(rets) else None,
        })

    return pd.DataFrame(rows)


def suggest_threshold(sub: pd.DataFrame, feature: str, horizon: int) -> str:
    """Suggest a threshold adjustment based on bucket win rates."""
    fwd_col = f"fwd_{horizon}d"
    valid = sub[[feature, fwd_col]].dropna()
    if len(valid) < 20:
        return "  (insufficient data for threshold suggestion)"

    # Find 75th percentile value — above this, wins tend to be higher
    p75 = valid[feature].quantile(0.75)
    p25 = valid[feature].quantile(0.25)

    high = valid[valid[feature] >= p75][fwd_col]
    low = valid[valid[feature] <= p25][fwd_col]

    if high.empty or low.empty:
        return "  (insufficient spread for threshold suggestion)"

    high_wr = (high > 0).mean() * 100
    low_wr = (low > 0).mean() * 100

    diff = high_wr - low_wr
    direction = "higher" if diff > 5 else "lower" if diff < -5 else "neutral"

    if direction == "higher":
        return f"  → Raise {feature} threshold to ≥{p75:.2g} (Q4 WR={high_wr:.0f}% vs Q1 WR={low_wr:.0f}%)"
    elif direction == "lower":
        return f"  → Lower {feature} threshold to ≤{p25:.2g} (Q1 WR={low_wr:.0f}% vs Q4 WR={high_wr:.0f}%)"
    else:
        return f"  → {feature} has low predictive value (Q1 WR={low_wr:.0f}% ≈ Q4 WR={high_wr:.0f}%)"


def analyze(picks_path: str, scanner_filter: str | None, horizon: int) -> None:
    df = load_picks(picks_path)
    print(f"\nLoaded {len(df)} picks from {picks_path}")

    scanners = [scanner_filter] if scanner_filter else sorted(df["scanner"].unique())

    for scanner in scanners:
        sub = df[df["scanner"] == scanner].copy()
        if sub.empty:
            print(f"\n⚠️  No picks for scanner: {scanner}")
            continue

        fwd_col = f"fwd_{horizon}d"
        valid = sub.dropna(subset=[fwd_col])
        if valid.empty:
            print(f"\n⚠️  No forward return data for {scanner} at {horizon}d horizon")
            continue

        wr = (valid[fwd_col] > 0).mean() * 100
        avg_ret = valid[fwd_col].mean() * 100

        print(f"\n{'='*64}")
        print(f"  Scanner: {scanner}")
        print(f"  Picks with {horizon}d return data: {len(valid)}  |  WR={wr:.1f}%  avg={avg_ret:+.2f}%")
        print(f"{'='*64}")

        # Feature correlations
        corr = correlation_table(sub, horizon)
        if not corr.empty:
            print(f"\n  Feature correlations with fwd_{horizon}d return (Spearman):")
            print(corr.to_string(index=False))

            # Bucket analysis for top 2 features by |spearman|
            top_features = corr.head(2)["feature"].tolist()
            for feat in top_features:
                buckets = bucket_analysis(sub, feat, horizon)
                if not buckets.empty:
                    print(f"\n  {feat} — quartile breakdown:")
                    print(buckets.to_string(index=False))
                    print(suggest_threshold(sub, feat, horizon))
        else:
            print(f"\n  No numeric features found with sufficient data.")

        # Priority breakdown
        if "priority" in sub.columns:
            print(f"\n  Win rate by priority:")
            for priority, grp in sub.groupby("priority"):
                g_valid = grp.dropna(subset=[fwd_col])
                if len(g_valid) >= 3:
                    g_wr = (g_valid[fwd_col] > 0).mean() * 100
                    g_avg = g_valid[fwd_col].mean() * 100
                    print(f"    {priority:10s}: n={len(g_valid):4d}  WR={g_wr:.1f}%  avg={g_avg:+.2f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze backtest picks for feature importance")
    parser.add_argument(
        "--path", default=None,
        help="Path to picks.csv (default: latest in results/backtest/)"
    )
    parser.add_argument("--scanner", default=None, help="Filter to a specific scanner")
    parser.add_argument(
        "--horizon", type=int, default=20,
        help="Forward return horizon in trading days (default: 20)"
    )
    args = parser.parse_args()

    if args.path:
        picks_path = args.path
    else:
        picks_path = str(find_latest_picks())
        print(f"Using latest backtest: {picks_path}")

    analyze(picks_path, args.scanner, args.horizon)


if __name__ == "__main__":
    main()

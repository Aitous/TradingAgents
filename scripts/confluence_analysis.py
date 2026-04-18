#!/usr/bin/env python3
"""Cross-scanner confluence analysis.

Finds dates where 2+ scanners fired on the same ticker on the same day, then
computes the win rate for confluence picks vs singleton picks. Reveals which
scanner pairs produce the best combined signal without any new research.

Usage:
    python scripts/confluence_analysis.py
    python scripts/confluence_analysis.py --path results/backtest/2026-04-16/picks.csv
    python scripts/confluence_analysis.py --horizon 20 --min-picks 5

Output:
    - Single-scanner WR baseline table
    - All scanner pairs with confluence WR, lift vs individual, and pick count
    - Ranked by confluence WR at the specified horizon
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_picks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def find_latest_picks(base_dir: str = "results/backtest") -> Path:
    runs = sorted(Path(base_dir).glob("*/picks.csv"))
    if not runs:
        raise FileNotFoundError(f"No picks.csv found under {base_dir}")
    return runs[-1]


def scanner_baseline(df: pd.DataFrame, horizon: int) -> dict:
    """Per-scanner WR and avg return at the given horizon."""
    fwd_col = f"fwd_{horizon}d"
    result = {}
    for scanner, grp in df.groupby("scanner"):
        valid = grp.dropna(subset=[fwd_col])
        if valid.empty:
            continue
        result[scanner] = {
            "picks": len(valid),
            "wr": round((valid[fwd_col] > 0).mean() * 100, 1),
            "avg": round(valid[fwd_col].mean() * 100, 2),
        }
    return result


def confluence_pairs(df: pd.DataFrame, horizon: int, min_picks: int) -> pd.DataFrame:
    """Compute WR for every (scanner_a, scanner_b) pair on same ticker+date."""
    fwd_col = f"fwd_{horizon}d"
    scanners = sorted(df["scanner"].unique())
    rows = []

    for s1, s2 in combinations(scanners, 2):
        # Get ticker+date sets for each scanner
        s1_picks = set(zip(df[df["scanner"] == s1]["date"], df[df["scanner"] == s1]["ticker"]))
        s2_picks = set(zip(df[df["scanner"] == s2]["date"], df[df["scanner"] == s2]["ticker"]))

        # Intersection = same ticker, same day
        overlap = s1_picks & s2_picks
        if len(overlap) < min_picks:
            continue

        # Get forward returns for the overlap picks (use s1's row — same fwd return)
        overlap_df = df[
            (df["scanner"] == s1) & df.apply(lambda r: (r["date"], r["ticker"]) in overlap, axis=1)
        ].dropna(subset=[fwd_col])

        if len(overlap_df) < min_picks:
            continue

        wr = (overlap_df[fwd_col] > 0).mean() * 100
        avg = overlap_df[fwd_col].mean() * 100

        # Individual WRs for lift calculation
        s1_wr = (df[df["scanner"] == s1].dropna(subset=[fwd_col])[fwd_col] > 0).mean() * 100
        s2_wr = (df[df["scanner"] == s2].dropna(subset=[fwd_col])[fwd_col] > 0).mean() * 100
        best_individual = max(s1_wr, s2_wr)
        lift = wr - best_individual

        rows.append(
            {
                "scanner_a": s1,
                "scanner_b": s2,
                "confluence_picks": len(overlap_df),
                f"confluence_wr_{horizon}d": round(wr, 1),
                f"confluence_avg_{horizon}d": round(avg, 2),
                "best_individual_wr": round(best_individual, 1),
                "wr_lift": round(lift, 1),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(f"confluence_wr_{horizon}d", ascending=False)


def analyze(picks_path: str, horizon: int, min_picks: int) -> None:
    df = load_picks(picks_path)
    print(f"\nLoaded {len(df)} picks from {picks_path}")
    print(f"Scanners: {', '.join(sorted(df['scanner'].unique()))}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    fwd_col = f"fwd_{horizon}d"
    if fwd_col not in df.columns:
        print(f"\n❌ Column '{fwd_col}' not found in picks.csv")
        return

    # Baseline per-scanner WR
    baseline = scanner_baseline(df, horizon)
    print(f"\n{'='*64}")
    print(f"  Single-scanner baseline at {horizon}d horizon")
    print(f"{'='*64}")
    base_df = pd.DataFrame(
        [
            {
                "scanner": k,
                "picks": v["picks"],
                f"wr_{horizon}d": v["wr"],
                f"avg_{horizon}d": v["avg"],
            }
            for k, v in sorted(baseline.items(), key=lambda x: -x[1]["wr"])
        ]
    )
    print(base_df.to_string(index=False))

    # Confluence pairs
    print(f"\n{'='*64}")
    print(f"  Confluence pairs (same ticker + same day, min {min_picks} picks)")
    print(f"{'='*64}")
    pairs = confluence_pairs(df, horizon, min_picks)

    if pairs.empty:
        print(f"\n  No confluence pairs with ≥{min_picks} overlapping picks found.")
        print("  Tip: Run with --min-picks 2 to see all pairs, or backtest more scanners together.")
    else:
        print(pairs.to_string(index=False))

        # Highlight pairs with positive lift
        positive_lift = pairs[pairs["wr_lift"] > 0]
        if not positive_lift.empty:
            print("\n  ⭐ Pairs with positive WR lift (confluence > best individual):")
            for _, row in positive_lift.iterrows():
                print(
                    f"    {row['scanner_a']} + {row['scanner_b']}: "
                    f"WR={row[f'confluence_wr_{horizon}d']:.0f}% "
                    f"(+{row['wr_lift']:.0f}pp lift, n={row['confluence_picks']})"
                )

    print()


def main():
    parser = argparse.ArgumentParser(description="Cross-scanner confluence analysis")
    parser.add_argument("--path", default=None, help="Path to picks.csv (default: latest)")
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Forward return horizon in trading days (default: 20)",
    )
    parser.add_argument(
        "--min-picks",
        type=int,
        default=5,
        help="Minimum overlapping picks to report a pair (default: 5)",
    )
    args = parser.parse_args()

    if args.path:
        picks_path = args.path
    else:
        picks_path = str(find_latest_picks())
        print(f"Using latest backtest: {picks_path}")

    analyze(picks_path, args.horizon, args.min_picks)


if __name__ == "__main__":
    main()

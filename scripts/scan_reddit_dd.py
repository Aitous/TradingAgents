#!/usr/bin/env python3
"""
Standalone Reddit DD Scanner
Scans Reddit for undiscovered high-quality Due Diligence posts and generates a markdown report.

Usage:
    python scripts/scan_reddit_dd.py [--hours HOURS] [--limit LIMIT] [--output FILE]

Examples:
    python scripts/scan_reddit_dd.py
    python scripts/scan_reddit_dd.py --hours 48 --limit 150
    python scripts/scan_reddit_dd.py --output reports/reddit_dd_2024_01_15.md
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.dataflows.reddit_api import get_reddit_undiscovered_dd
from langchain_openai import ChatOpenAI


def main():
    parser = argparse.ArgumentParser(description='Scan Reddit for high-quality DD posts')
    parser.add_argument('--hours', type=int, default=72, help='Hours to look back (default: 72)')
    parser.add_argument('--limit', type=int, default=100, help='Number of posts to scan (default: 100)')
    parser.add_argument('--top', type=int, default=15, help='Number of top DD to include (default: 15)')
    parser.add_argument('--output', type=str, help='Output markdown file (default: reports/reddit_dd_YYYY_MM_DD.md)')
    parser.add_argument('--min-score', type=int, default=55, help='Minimum quality score (default: 55)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='LLM model to use (default: gpt-4o-mini)')
    parser.add_argument('--temperature', type=float, default=0, help='LLM temperature (default: 0)')
    parser.add_argument('--comments', type=int, default=10, help='Number of top comments to include (default: 10)')

    args = parser.parse_args()

    # Setup output file
    if args.output:
        output_file = args.output
    else:
        # Create reports directory if it doesn't exist
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        output_file = reports_dir / f"reddit_dd_{timestamp}.md"

    print("=" * 70)
    print("📊 REDDIT DD SCANNER")
    print("=" * 70)
    print(f"Lookback: {args.hours} hours")
    print(f"Scan limit: {args.limit} posts")
    print(f"Top results: {args.top}")
    print(f"Min quality score: {args.min_score}")
    print(f"LLM model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {output_file}")
    print("=" * 70)
    print()

    # Initialize LLM
    print("Initializing LLM...")
    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Scan Reddit
    print(f"\n🔍 Scanning Reddit (last {args.hours} hours)...\n")

    dd_report = get_reddit_undiscovered_dd(
        lookback_hours=args.hours,
        scan_limit=args.limit,
        top_n=args.top,
        num_comments=args.comments,
        llm_evaluator=llm
    )

    # Add header with metadata
    header = f"""# 📊 Reddit DD Scanner Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Lookback Period:** {args.hours} hours
**Posts Scanned:** {args.limit}
**Minimum Quality Score:** {args.min_score}/100

---

"""

    full_report = header + dd_report

    # Save to file
    with open(output_file, 'w') as f:
        f.write(full_report)

    print("\n" + "=" * 70)
    print(f"✅ Report saved to: {output_file}")
    print("=" * 70)

    # Print summary
    print("\n📈 SUMMARY:")

    # Count quality posts by parsing the report
    import re
    quality_match = re.search(r'\*\*High Quality:\*\* (\d+) DD posts', dd_report)
    scanned_match = re.search(r'\*\*Scanned:\*\* (\d+) posts', dd_report)

    if scanned_match and quality_match:
        scanned = int(scanned_match.group(1))
        quality = int(quality_match.group(1))
        print(f"  • Posts scanned: {scanned}")
        print(f"  • Quality DD found: {quality}")
        if scanned > 0:
            print(f"  • Quality rate: {(quality/scanned)*100:.1f}%")

    # Extract tickers
    ticker_matches = re.findall(r'\*\*Ticker:\*\* \$([A-Z]+)', dd_report)
    if ticker_matches:
        unique_tickers = list(set(ticker_matches))
        print(f"  • Tickers mentioned: {', '.join(['$' + t for t in unique_tickers])}")

    print()
    print("💡 TIP: Review the report and investigate promising opportunities!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

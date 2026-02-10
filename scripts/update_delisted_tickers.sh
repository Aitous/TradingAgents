#!/bin/bash
# Script to extract consistently failing tickers from the delisted cache
# These are candidates for adding to PERMANENTLY_DELISTED after manual verification

CACHE_FILE="data/delisted_cache.json"
REVIEW_FILE="data/delisted_review.txt"

echo "Analyzing delisted cache for consistently failing tickers..."

if [ ! -f "$CACHE_FILE" ]; then
    echo "No delisted cache found at $CACHE_FILE"
    echo "Run discovery flow at least once to populate the cache."
    exit 0
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed."
    echo "Install it with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 1
fi

# Extract tickers with high fail counts (3+ failures across multiple days)
echo ""
echo "Tickers that have failed 3+ times:"
echo "=================================="
jq -r 'to_entries[] | select(.value.fail_count >= 3) | "\(.key): \(.value.fail_count) failures across \(.value.fail_dates | length) days - \(.value.reason)"' "$CACHE_FILE"

echo ""
echo "---"
echo "Review the tickers above and verify their status using:"
echo "  1. Yahoo Finance: https://finance.yahoo.com/quote/TICKER"
echo "  2. SEC EDGAR: https://www.sec.gov/cgi-bin/browse-edgar"
echo "  3. Google search: 'TICKER stock delisted'"
echo ""
echo "For CONFIRMED permanent delistings, add them to PERMANENTLY_DELISTED in:"
echo "  tradingagents/graph/discovery_graph.py"
echo ""
echo "Detailed review list has been exported to: $REVIEW_FILE"

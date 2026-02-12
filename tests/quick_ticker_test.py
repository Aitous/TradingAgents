"""
Quick ticker matcher validation
"""
from tradingagents.dataflows.discovery.ticker_matcher import match_company_to_ticker, load_ticker_universe

# Load universe
print("Loading ticker universe...")
universe = load_ticker_universe()
print(f"Loaded {len(universe)} tickers\n")

# Test cases
tests = [
    ("Apple Inc", "AAPL"),
    ("MICROSOFT CORP", "MSFT"),
    ("Amazon.com, Inc.", "AMZN"),
    ("TESLA INC", "TSLA"),
    ("META PLATFORMS INC", "META"),
    ("NVIDIA CORPORATION", "NVDA"),
]

print("Testing ticker matching:")
for company, expected in tests:
    result = match_company_to_ticker(company)
    status = "✓" if result and result.startswith(expected[:3]) else "✗"
    print(f"{status} '{company}' -> {result} (expected {expected})")

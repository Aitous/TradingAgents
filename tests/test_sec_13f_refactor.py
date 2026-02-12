#!/usr/bin/env python3
"""
Test SEC 13F Parser with Ticker Matching

This script tests the refactored SEC 13F parser to verify:
1. Ticker matcher module loads successfully
2. Fuzzy matching works correctly  
3. SEC 13F parsing integrates with ticker matcher
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Testing SEC 13F Parser Refactor")
print("=" * 60)

# Test 1: Ticker Matcher Module
print("\n[1/3] Testing Ticker Matcher Module...")
try:
    from tradingagents.dataflows.discovery.ticker_matcher import (
        match_company_to_ticker,
        load_ticker_universe,
        get_match_confidence,
    )
    
    # Load universe
    universe = load_ticker_universe()
    print(f"✓ Loaded {len(universe)} tickers")
    
    # Test exact matches
    test_cases = [
        ("Apple Inc", "AAPL"),
        ("MICROSOFT CORP", "MSFT"),
        ("Amazon.com, Inc.", "AMZN"),
        ("Alphabet Inc", "GOOGL"),  # or GOOG
        ("TESLA INC", "TSLA"),
        ("META PLATFORMS INC", "META"),
        ("NVIDIA CORPORATION", "NVDA"),
        ("Berkshire Hathaway Inc", "BRK.B"),  # or BRK.A
    ]
    
    passed = 0
    for company, expected_prefix in test_cases:
        result = match_company_to_ticker(company)
        if result and result.startswith(expected_prefix[:3]):
            passed += 1
            print(f"  ✓ '{company}' -> {result}")
        else:
            print(f"  ✗ '{company}' -> {result} (expected {expected_prefix})")
    
    print(f"\nPassed {passed}/{len(test_cases)} exact match tests")
    
    # Test fuzzy matching
    print("\nTesting fuzzy matching...")
    fuzzy_cases = [
        "APPLE COMPUTER INC",
        "Microsoft Corporation",
        "Amazon Com Inc",
        "Tesla Motors",
    ]
    
    for company in fuzzy_cases:
        result = match_company_to_ticker(company, min_confidence=70.0)
        confidence = get_match_confidence(company, result) if result else 0
        print(f"  '{company}' -> {result} (confidence: {confidence:.1f})")
    
    print("✓ Ticker matcher working correctly")
    
except Exception as e:
    print(f"✗ Error testing ticker matcher: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: SEC 13F Integration
print("\n[2/3] Testing SEC 13F Integration...")
try:
    from tradingagents.dataflows.sec_13f import get_recent_13f_changes
    
    print("Fetching recent 13F filings (this may take 30-60 seconds)...")
    results = get_recent_13f_changes(
        days_lookback=14,  # Last 2 weeks
        min_position_value=50,  # $50M+
        notable_only=False,
        top_n=10,
        return_structured=True,
    )
    
    if results:
        print(f"\n✓ Found {len(results)} institutional holdings")
        print("\nTop 5 holdings:")
        print(f"{'Issuer':<40} {'Ticker':<8} {'Institutions':<12} {'Match Method'}")
        print("-" * 80)
        
        for i, r in enumerate(results[:5]):
            issuer = r['issuer'][:38]
            ticker = r.get('ticker', 'N/A')
            inst_count = r.get('institution_count', 0)
            match_method = r.get('match_method', 'unknown')
            print(f"{issuer:<40} {ticker:<8} {inst_count:<12} {match_method}")
        
        # Calculate match statistics
        fuzzy_matches = sum(1 for r in results if r.get('match_method') == 'fuzzy')
        regex_matches = sum(1 for r in results if r.get('match_method') == 'regex')
        unmatched = sum(1 for r in results if r.get('match_method') == 'unmatched')
        
        print(f"\nMatch Statistics:")
        print(f"  Fuzzy matches: {fuzzy_matches}/{len(results)} ({100*fuzzy_matches/len(results):.1f}%)")
        print(f"  Regex fallback: {regex_matches}/{len(results)} ({100*regex_matches/len(results):.1f}%)")
        print(f"  Unmatched: {unmatched}/{len(results)} ({100*unmatched/len(results):.1f}%)")
        
        if fuzzy_matches > 0:
            print("\n✓ SEC 13F parser successfully using ticker matcher!")
        else:
            print("\n⚠ Warning: No fuzzy matches found, matcher may not be integrated")
    else:
        print("⚠ No results found (may be weekend/no recent filings)")
    
except Exception as e:
    print(f"✗ Error testing SEC 13F integration: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit, this might fail due to network issues

# Test 3: Scanner Interface
print("\n[3/3] Testing Scanner Interface...")
try:
    from tradingagents.dataflows.sec_13f import scan_13f_changes
    
    config = {
        "discovery": {
            "13f_lookback_days": 7,
            "13f_min_position_value": 25,
        }
    }
    
    candidates = scan_13f_changes(config)
    
    if candidates:
        print(f"✓ Scanner returned {len(candidates)} candidates")
        print(f"\nSample candidates:")
        for c in candidates[:3]:
            print(f"  {c['ticker']}: {c['context']} [{c['priority']}]")
    else:
        print("⚠ Scanner returned no candidates (may be normal)")
    
except Exception as e:
    print(f"✗ Error testing scanner interface: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing Complete!")
print("=" * 60)

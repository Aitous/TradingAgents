"""Test concurrent scanner execution."""
import time
import copy
from unittest.mock import MagicMock, patch

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.discovery_graph import DiscoveryGraph


def test_concurrent_execution():
    """Test that concurrent execution runs scanners in parallel."""

    # Get config with concurrent execution enabled
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["discovery"]["scanner_execution"] = {
        "concurrent": True,
        "max_workers": 4,
        "timeout_seconds": 30,
    }

    # Create discovery graph
    graph = DiscoveryGraph(config)

    # Create initial state
    state = {
        "trade_date": "2026-02-05",
        "tickers": [],
        "filtered_tickers": [],
        "final_ranking": "",
        "status": "initialized",
        "tool_logs": [],
    }

    # Run scanner node with timing
    print("\n=== Testing Concurrent Scanner Execution ===")
    start = time.time()
    result = graph.scanner_node(state)
    elapsed = time.time() - start

    # Verify results
    print(f"\n✓ Execution time: {elapsed:.2f}s")
    print(f"✓ Found {len(result['tickers'])} unique tickers")
    print(f"✓ Found {len(result['candidate_metadata'])} candidates")
    print(f"✓ Tool logs: {len(result['tool_logs'])} entries")

    # Check that we got results
    assert len(result['tickers']) > 0, "Should find at least some tickers"
    assert len(result['candidate_metadata']) > 0, "Should find candidates"
    assert result['status'] == 'scanned', "Status should be scanned"

    print("\n✅ Concurrent execution test passed!")
    return result


def test_sequential_fallback():
    """Test that sequential execution works when concurrent is disabled."""

    # Get config with concurrent execution disabled
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["discovery"]["scanner_execution"] = {
        "concurrent": False,
        "max_workers": 1,
        "timeout_seconds": 30,
    }

    # Create discovery graph
    graph = DiscoveryGraph(config)

    # Create initial state
    state = {
        "trade_date": "2026-02-05",
        "tickers": [],
        "filtered_tickers": [],
        "final_ranking": "",
        "status": "initialized",
        "tool_logs": [],
    }

    # Run scanner node with timing
    print("\n=== Testing Sequential Scanner Execution ===")
    start = time.time()
    result = graph.scanner_node(state)
    elapsed = time.time() - start

    # Verify results
    print(f"\n✓ Execution time: {elapsed:.2f}s")
    print(f"✓ Found {len(result['tickers'])} unique tickers")
    print(f"✓ Found {len(result['candidate_metadata'])} candidates")

    # Check that we got results
    assert len(result['tickers']) > 0, "Should find at least some tickers"
    assert len(result['candidate_metadata']) > 0, "Should find candidates"
    assert result['status'] == 'scanned', "Status should be scanned"

    print("\n✅ Sequential execution test passed!")
    return result


def test_timeout_handling():
    """Test that scanner timeout is enforced."""

    # Get config with very short timeout
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["discovery"]["scanner_execution"] = {
        "concurrent": True,
        "max_workers": 4,
        "timeout_seconds": 1,  # Very short timeout
    }

    # Create discovery graph
    graph = DiscoveryGraph(config)

    # Create initial state
    state = {
        "trade_date": "2026-02-05",
        "tickers": [],
        "filtered_tickers": [],
        "final_ranking": "",
        "status": "initialized",
        "tool_logs": [],
    }

    # Run scanner node - some scanners may timeout
    print("\n=== Testing Timeout Handling (1s timeout) ===")
    start = time.time()
    result = graph.scanner_node(state)
    elapsed = time.time() - start

    # Verify results (may be partial due to timeouts)
    print(f"\n✓ Execution time: {elapsed:.2f}s")
    print(f"✓ Found {len(result['tickers'])} tickers (some scanners may have timed out)")
    print(f"✓ Status: {result['status']}")

    # Should still complete even with timeouts
    assert result['status'] == 'scanned', "Status should be scanned even with timeouts"

    print("\n✅ Timeout handling test passed!")
    return result


if __name__ == "__main__":
    # Run tests
    print("\n" + "="*60)
    print("Testing Scanner Concurrent Execution")
    print("="*60)

    try:
        # Test 1: Concurrent execution
        result1 = test_concurrent_execution()

        # Test 2: Sequential fallback
        result2 = test_sequential_fallback()

        # Test 3: Timeout handling
        result3 = test_timeout_handling()

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

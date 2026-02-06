#!/usr/bin/env python3
"""Quick verification that concurrent scanner execution works."""
import time
import copy
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.discovery_graph import DiscoveryGraph


def compare_execution_modes():
    """Compare concurrent vs sequential execution."""

    print("\n" + "="*60)
    print("Concurrent Scanner Execution Verification")
    print("="*60)

    # Test 1: Concurrent execution
    print("\n1️⃣  Testing CONCURRENT execution...")
    config_concurrent = copy.deepcopy(DEFAULT_CONFIG)
    config_concurrent["discovery"]["scanner_execution"] = {
        "concurrent": True,
        "max_workers": 8,
        "timeout_seconds": 30,
    }

    graph_concurrent = DiscoveryGraph(config_concurrent)
    state = {
        "trade_date": "2026-02-05",
        "tickers": [],
        "tool_logs": [],
    }

    start = time.time()
    result_concurrent = graph_concurrent.scanner_node(state)
    time_concurrent = time.time() - start

    print(f"\n   ⏱️  Concurrent time: {time_concurrent:.2f}s")
    print(f"   📊 Candidates found: {len(result_concurrent['candidate_metadata'])}")

    # Test 2: Sequential execution
    print("\n2️⃣  Testing SEQUENTIAL execution...")
    config_sequential = copy.deepcopy(DEFAULT_CONFIG)
    config_sequential["discovery"]["scanner_execution"] = {
        "concurrent": False,
        "max_workers": 1,
        "timeout_seconds": 30,
    }

    graph_sequential = DiscoveryGraph(config_sequential)
    state = {
        "trade_date": "2026-02-05",
        "tickers": [],
        "tool_logs": [],
    }

    start = time.time()
    result_sequential = graph_sequential.scanner_node(state)
    time_sequential = time.time() - start

    print(f"\n   ⏱️  Sequential time: {time_sequential:.2f}s")
    print(f"   📊 Candidates found: {len(result_sequential['candidate_metadata'])}")

    # Compare
    improvement = ((time_sequential - time_concurrent) / time_sequential) * 100

    print("\n" + "="*60)
    print("📊 Performance Comparison")
    print("="*60)
    print(f"Concurrent:  {time_concurrent:.2f}s ({len(result_concurrent['tickers'])} tickers)")
    print(f"Sequential:  {time_sequential:.2f}s ({len(result_sequential['tickers'])} tickers)")
    print(f"Improvement: {improvement:.1f}% faster ⚡")
    print("="*60)

    return {
        "concurrent_time": time_concurrent,
        "sequential_time": time_sequential,
        "improvement_pct": improvement,
        "concurrent_candidates": len(result_concurrent['candidate_metadata']),
        "sequential_candidates": len(result_sequential['candidate_metadata']),
    }


if __name__ == "__main__":
    results = compare_execution_modes()

    # Verify improvement
    if results["improvement_pct"] > 15:
        print(f"\n✅ SUCCESS: Concurrent execution is {results['improvement_pct']:.1f}% faster!")
    else:
        print(f"\n⚠️  WARNING: Only {results['improvement_pct']:.1f}% improvement")

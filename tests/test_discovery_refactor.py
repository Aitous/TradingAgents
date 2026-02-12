#!/usr/bin/env python3
"""
Test script to verify DiscoveryGraph refactoring.
Tests: LLM Factory, TraditionalScanner, CandidateFilter, CandidateRanker
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_llm_factory():
    """Test LLM factory initialization."""
    print("\n=== Testing LLM Factory ===")
    try:
        from tradingagents.utils.llm_factory import create_llms

        # Mock API key
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

        config = {
            "llm_provider": "openai",
            "deep_think_llm": "gpt-4",
            "quick_think_llm": "gpt-3.5-turbo"
        }

        deep_llm, quick_llm = create_llms(config)

        assert deep_llm is not None, "Deep LLM should be initialized"
        assert quick_llm is not None, "Quick LLM should be initialized"

        print("✅ LLM Factory: Successfully creates LLMs")
        return True

    except Exception as e:
        print(f"❌ LLM Factory: Failed - {e}")
        return False

def test_traditional_scanner():
    """Test TraditionalScanner class."""
    print("\n=== Testing TraditionalScanner ===")
    try:
        from unittest.mock import MagicMock

        from tradingagents.dataflows.discovery.scanners import TraditionalScanner

        config = {"discovery": {}}
        mock_llm = MagicMock()
        mock_executor = MagicMock()

        scanner = TraditionalScanner(config, mock_llm, mock_executor)

        assert hasattr(scanner, 'scan'), "Scanner should have scan method"
        assert scanner.execute_tool == mock_executor, "Should store executor"

        print("✅ TraditionalScanner: Successfully initialized")
        return True

    except Exception as e:
        print(f"❌ TraditionalScanner: Failed - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_candidate_filter():
    """Test CandidateFilter class."""
    print("\n=== Testing CandidateFilter ===")
    try:
        from unittest.mock import MagicMock

        from tradingagents.dataflows.discovery.filter import CandidateFilter

        config = {"discovery": {}}
        mock_executor = MagicMock()

        filter_obj = CandidateFilter(config, mock_executor)

        assert hasattr(filter_obj, 'filter'), "Filter should have filter method"
        assert filter_obj.execute_tool == mock_executor, "Should store executor"

        print("✅ CandidateFilter: Successfully initialized")
        return True

    except Exception as e:
        print(f"❌ CandidateFilter: Failed - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_candidate_ranker():
    """Test CandidateRanker class."""
    print("\n=== Testing CandidateRanker ===")
    try:
        from unittest.mock import MagicMock

        from tradingagents.dataflows.discovery.ranker import CandidateRanker

        config = {"discovery": {}}
        mock_llm = MagicMock()
        mock_analytics = MagicMock()

        ranker = CandidateRanker(config, mock_llm, mock_analytics)

        assert hasattr(ranker, 'rank'), "Ranker should have rank method"
        assert ranker.llm == mock_llm, "Should store LLM"

        print("✅ CandidateRanker: Successfully initialized")
        return True

    except Exception as e:
        print(f"❌ CandidateRanker: Failed - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_discovery_graph_import():
    """Test that DiscoveryGraph still imports correctly."""
    print("\n=== Testing DiscoveryGraph Import ===")
    try:
        from tradingagents.graph.discovery_graph import DiscoveryGraph

        # Mock API key
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

        config = {
            "llm_provider": "openai",
            "deep_think_llm": "gpt-4",
            "quick_think_llm": "gpt-3.5-turbo",
            "backend_url": "https://api.openai.com/v1",
            "discovery": {}
        }

        graph = DiscoveryGraph(config=config)

        assert hasattr(graph, 'deep_thinking_llm'), "Should have deep LLM"
        assert hasattr(graph, 'quick_thinking_llm'), "Should have quick LLM"
        assert hasattr(graph, 'analytics'), "Should have analytics"
        assert hasattr(graph, 'graph'), "Should have graph"

        print("✅ DiscoveryGraph: Successfully initialized with refactored components")
        return True

    except Exception as e:
        print(f"❌ DiscoveryGraph: Failed - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trading_graph_import():
    """Test that TradingAgentsGraph still imports correctly."""
    print("\n=== Testing TradingAgentsGraph Import ===")
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        # Mock API key
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

        config = {
            "llm_provider": "openai",
            "deep_think_llm": "gpt-4",
            "quick_think_llm": "gpt-3.5-turbo",
            "project_dir": str(project_root),
            "enable_memory": False
        }

        graph = TradingAgentsGraph(config=config)

        assert hasattr(graph, 'deep_thinking_llm'), "Should have deep LLM"
        assert hasattr(graph, 'quick_thinking_llm'), "Should have quick LLM"

        print("✅ TradingAgentsGraph: Successfully initialized with LLM factory")
        return True

    except Exception as e:
        print(f"❌ TradingAgentsGraph: Failed - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils():
    """Test utility functions."""
    print("\n=== Testing Utilities ===")
    try:
        from tradingagents.dataflows.discovery.utils import (
            extract_technical_summary,
            is_valid_ticker,
        )

        # Test ticker validation
        assert is_valid_ticker("AAPL") == True, "AAPL should be valid"
        assert is_valid_ticker("AAPL.WS") == False, "Warrant should be invalid"
        assert is_valid_ticker("AAPL-RT") == False, "Rights should be invalid"

        # Test technical summary extraction
        tech_report = "RSI Value: 45.5"
        summary = extract_technical_summary(tech_report)
        assert "RSI:45" in summary or "RSI:46" in summary, "Should extract RSI"

        print("✅ Utils: All utility functions work correctly")
        return True

    except Exception as e:
        print(f"❌ Utils: Failed - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DISCOVERY GRAPH REFACTORING VERIFICATION")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("LLM Factory", test_llm_factory()))
    results.append(("Traditional Scanner", test_traditional_scanner()))
    results.append(("Candidate Filter", test_candidate_filter()))
    results.append(("Candidate Ranker", test_candidate_ranker()))
    results.append(("Utils", test_utils()))
    results.append(("DiscoveryGraph", test_discovery_graph_import()))
    results.append(("TradingAgentsGraph", test_trading_graph_import()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All refactoring tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

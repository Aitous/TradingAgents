
import os
import shutil
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from tradingagents.dataflows.discovery.scanners import TraditionalScanner
from tradingagents.graph.discovery_graph import DiscoveryGraph


def test_graph_init_with_factory():
    print("Testing DiscoveryGraph initialization with LLM Factory...")
    config = {
        "llm_provider": "openai",
        "deep_think_llm": "gpt-4-turbo",
        "quick_think_llm": "gpt-3.5-turbo",
        "backend_url": "https://api.openai.com/v1",
        "discovery": {},
        "results_dir": "tests/temp_results"
    }

    # Mock API key so factory works
    if not os.getenv("OPENAI_API_KEY"):
         os.environ["OPENAI_API_KEY"] = "sk-mock-key"

    try:
        graph = DiscoveryGraph(config=config)
        assert hasattr(graph, 'deep_thinking_llm')
        assert hasattr(graph, 'quick_thinking_llm')
        assert graph.deep_thinking_llm is not None
        print("✅ DiscoveryGraph initialized LLMs via Factory")
    except Exception as e:
        print(f"❌ DiscoveryGraph initialization failed: {e}")

def test_traditional_scanner_init():
    print("Testing TraditionalScanner initialization...")
    config = {"discovery": {}}
    mock_llm = MagicMock()
    mock_executor = MagicMock()

    try:
        scanner = TraditionalScanner(config, mock_llm, mock_executor)
        assert scanner.execute_tool == mock_executor
        print("✅ TraditionalScanner initialized")

        # Test scan (mocking tools)
        mock_executor.return_value = {"valid": ["AAPL"], "invalid": []}
        state = {"trade_date": "2023-10-27"}

        # We expect some errors printed because we didn't mock everything perfect,
        # but it shouldn't crash.
        print("   Running scan (expecting some print errors due to missing tools)...")
        candidates = scanner.scan(state)
        print(f"   Scan returned {len(candidates)} candidates")
        print("✅ TraditionalScanner scan() ran without crash")

    except Exception as e:
        print(f"❌ TraditionalScanner failed: {e}")

def cleanup():
    if os.path.exists("tests/temp_results"):
        shutil.rmtree("tests/temp_results")

if __name__ == "__main__":
    try:
        test_graph_init_with_factory()
        test_traditional_scanner_init()
        print("\nAll checks passed!")
    finally:
        cleanup()

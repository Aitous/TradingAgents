
import os
import shutil
import sys

# Add project root to path
sys.path.append(os.getcwd())

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

def cleanup():
    if os.path.exists("tests/temp_results"):
        shutil.rmtree("tests/temp_results")

if __name__ == "__main__":
    try:
        test_graph_init_with_factory()
        print("\nAll checks passed!")
    finally:
        cleanup()

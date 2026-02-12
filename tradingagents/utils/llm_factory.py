import os
from typing import Any, Dict, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


def create_llms(config: Dict[str, Any]) -> Tuple[BaseChatModel, BaseChatModel]:
    """
    Factory to create deep and quick thinking LLMs based on configuration.

    Args:
        config: Configuration dictionary containing keys:
            - llm_provider: 'openai', 'anthropic', 'google', 'ollama', or 'openrouter'
            - deep_think_llm: Model name for complex reasoning
            - quick_think_llm: Model name for simple tasks

    Returns:
        Tuple containing (deep_thinking_llm, quick_thinking_llm)

    Raises:
        ValueError: If provider is unsupported or API keys are missing.
    """
    provider = config.get("llm_provider", "openai").lower()

    if provider in ["openai", "ollama", "openrouter"]:
        api_key = os.getenv("OPENAI_API_KEY")
        # For Ollama (local), API key might not be needed, but usually langgraph expects it or base_url
        # If openrouter, it uses openai compatible interface

        deep_llm = ChatOpenAI(model=config["deep_think_llm"], api_key=api_key)
        quick_llm = ChatOpenAI(model=config["quick_think_llm"], api_key=api_key)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            # Try to warn but proceed (library might raise)
            logger.warning("ANTHROPIC_API_KEY not found in environment.")

        deep_llm = ChatAnthropic(model=config["deep_think_llm"], api_key=api_key)
        quick_llm = ChatAnthropic(model=config["quick_think_llm"], api_key=api_key)

    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. Please add it to your .env file."
            )

        deep_llm = ChatGoogleGenerativeAI(model=config["deep_think_llm"], google_api_key=api_key)
        quick_llm = ChatGoogleGenerativeAI(model=config["quick_think_llm"], google_api_key=api_key)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return deep_llm, quick_llm

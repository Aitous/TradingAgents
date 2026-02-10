import os
from typing import Any, Optional

from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Centralized configuration management.
    Merges environment variables with default configuration.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._defaults = DEFAULT_CONFIG
        self._env_cache = {}

    def _get_env(self, key: str, default: Any = None) -> Any:
        """Helper to get env var with optional default from config dictionary."""
        val = os.getenv(key)
        if val is not None:
            return val
        return default

    # --- API Keys ---

    @property
    def openai_api_key(self) -> Optional[str]:
        return self._get_env("OPENAI_API_KEY")

    @property
    def alpha_vantage_api_key(self) -> Optional[str]:
        return self._get_env("ALPHA_VANTAGE_API_KEY")

    @property
    def finnhub_api_key(self) -> Optional[str]:
        return self._get_env("FINNHUB_API_KEY")

    @property
    def tradier_api_key(self) -> Optional[str]:
        return self._get_env("TRADIER_API_KEY")

    @property
    def fmp_api_key(self) -> Optional[str]:
        return self._get_env("FMP_API_KEY")

    @property
    def reddit_client_id(self) -> Optional[str]:
        return self._get_env("REDDIT_CLIENT_ID")

    @property
    def reddit_client_secret(self) -> Optional[str]:
        return self._get_env("REDDIT_CLIENT_SECRET")

    @property
    def reddit_user_agent(self) -> str:
        return self._get_env("REDDIT_USER_AGENT", "TradingAgents/1.0")

    @property
    def twitter_bearer_token(self) -> Optional[str]:
        return self._get_env("TWITTER_BEARER_TOKEN")

    @property
    def serper_api_key(self) -> Optional[str]:
        return self._get_env("SERPER_API_KEY")

    @property
    def gemini_api_key(self) -> Optional[str]:
        return self._get_env("GEMINI_API_KEY")

    # --- Paths and Settings ---

    @property
    def results_dir(self) -> str:
        return self._defaults.get("results_dir", "./results")

    @property
    def user_workspace(self) -> str:
        return self._get_env("USER_WORKSPACE", self._defaults.get("project_dir"))

    # --- Methods ---

    def validate_key(self, key_property: str, service_name: str) -> str:
        """
        Validate that a specific API key property is set.
        Returns the key if valid, raises ValueError otherwise.
        """
        key = getattr(self, key_property)
        if not key:
            raise ValueError(
                f"{service_name} API Key not found. Please set correct environment variable."
            )
        return key

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        Checks properties first, then defaults.
        """
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val

        return self._defaults.get(key, default)


# Global config instance
config = Config()

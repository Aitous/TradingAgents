
import os
from unittest.mock import patch

import pytest

from tradingagents.config import Config


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "ALPHA_VANTAGE_API_KEY": "test-alpha-key",
        "FINNHUB_API_KEY": "test-finnhub-key",
        "TRADIER_API_KEY": "test-tradier-key",
        "GOOGLE_API_KEY": "test-google-key",
        "REDDIT_CLIENT_ID": "test-reddit-id",
        "REDDIT_CLIENT_SECRET": "test-reddit-secret",
        "TWITTER_BEARER_TOKEN": "test-twitter-token"
    }, clear=True):
        yield

@pytest.fixture
def mock_config(mock_env_vars):
    """Return a Config instance with mocked env vars."""
    # Reset singleton
    Config._instance = None
    return Config()

@pytest.fixture
def sample_stock_data():
    """Return a sample DataFrame for technical analysis."""
    import pandas as pd
    data = {
        "close": [100, 102, 101, 103, 105, 108, 110, 109, 112, 115],
        "high": [105, 106, 105, 107, 108, 112, 115, 113, 116, 118],
        "low": [95, 98, 99, 100, 102, 105, 108, 106, 108, 111],
        "volume": [1000] * 10
    }
    return pd.DataFrame(data)

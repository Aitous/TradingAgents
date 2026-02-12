
import os
from unittest.mock import patch

import pytest

from tradingagents.config import Config


class TestConfig:
    def test_singleton(self):
        Config._instance = None
        c1 = Config()
        c2 = Config()
        assert c1 is c2

    def test_validate_key_success(self, mock_env_vars):
        Config._instance = None
        config = Config()
        key = config.validate_key("openai_api_key", "OpenAI")
        assert key == "test-openai-key"

    def test_validate_key_failure(self):
        Config._instance = None
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ValueError) as excinfo:
                config.validate_key("openai_api_key", "OpenAI")
            assert "OpenAI API Key not found" in str(excinfo.value)

    def test_get_method(self):
        Config._instance = None
        config = Config()
        # Test getting real property
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            assert config.get("openai_api_key") == "test-key"

        # Test getting default value
        assert config.get("results_dir") == "./results"

        # Test fallback to provided default
        assert config.get("non_existent_key", "default") == "default"

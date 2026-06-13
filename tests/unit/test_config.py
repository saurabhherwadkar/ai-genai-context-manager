"""Unit tests for configuration loading and validation."""

import os
from pathlib import Path

import pytest
import yaml

from context_manager.config import (
    ProviderConfig,
    SecurityConfig,
    TokenCountingConfig,
    TrimmingConfig,
    load_config,
    load_config_from_env,
)
from context_manager.exceptions import ConfigurationError


class TestProviderConfig:
    """Tests for the ProviderConfig model."""

    def test_default_values(self) -> None:
        """Verify ProviderConfig has sensible defaults."""
        # Create with defaults
        config = ProviderConfig()
        # Assert default values
        assert config.name == "openai"
        assert config.model == "gpt-4o"
        assert config.max_context_tokens == 128_000
        assert config.reserved_response_tokens == 4096
        assert config.timeout_seconds == 30
        assert config.max_retries == 3

    def test_invalid_max_context_tokens(self) -> None:
        """Verify validation rejects non-positive max_context_tokens."""
        # Attempt to create with invalid value
        with pytest.raises((ValueError, Exception), match="must be"):
            ProviderConfig(max_context_tokens=0)

    def test_invalid_reserved_response_tokens(self) -> None:
        """Verify validation rejects negative reserved_response_tokens."""
        # Attempt to create with negative value
        with pytest.raises((ValueError, Exception), match="must be"):
            ProviderConfig(reserved_response_tokens=-1)


class TestTokenCountingConfig:
    """Tests for the TokenCountingConfig model."""

    def test_default_values(self) -> None:
        """Verify TokenCountingConfig has sensible defaults."""
        # Create with defaults
        config = TokenCountingConfig()
        # Assert defaults
        assert config.strategy == "tiktoken"
        assert config.encoding == "cl100k_base"
        assert config.estimator_ratio == 4.0

    def test_invalid_estimator_ratio(self) -> None:
        """Verify validation rejects non-positive estimator_ratio."""
        # Attempt to create with invalid ratio
        with pytest.raises((ValueError, Exception), match="must be"):
            TokenCountingConfig(estimator_ratio=0.0)


class TestTrimmingConfig:
    """Tests for the TrimmingConfig model."""

    def test_default_values(self) -> None:
        """Verify TrimmingConfig has sensible defaults."""
        # Create with defaults
        config = TrimmingConfig()
        # Assert defaults
        assert config.strategy == "fifo"
        assert config.window_size == 50
        assert config.preserve_system_message is True

    def test_invalid_window_size(self) -> None:
        """Verify validation rejects non-positive window_size."""
        # Attempt to create with invalid size
        with pytest.raises((ValueError, Exception), match="must be"):
            TrimmingConfig(window_size=0)


class TestSecurityConfig:
    """Tests for the SecurityConfig model."""

    def test_default_values(self) -> None:
        """Verify SecurityConfig has sensible defaults."""
        # Create with defaults
        config = SecurityConfig()
        # Assert defaults
        assert config.max_message_length == 100_000
        assert config.max_messages == 1000
        assert config.sanitize_input is True

    def test_invalid_max_message_length(self) -> None:
        """Verify validation rejects non-positive max_message_length."""
        # Attempt to create with zero length
        with pytest.raises((ValueError, Exception), match="must be"):
            SecurityConfig(max_message_length=0)

    def test_invalid_max_messages(self) -> None:
        """Verify validation rejects non-positive max_messages."""
        # Attempt to create with zero messages
        with pytest.raises((ValueError, Exception), match="must be"):
            SecurityConfig(max_messages=0)


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Verify loading a valid YAML config file."""
        # Write a valid config to a temp file
        config_data = {"environment": "test", "provider": {"name": "anthropic"}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        # Load the config
        config = load_config(config_file)
        # Assert values are loaded correctly
        assert config.environment == "test"
        assert config.provider.name == "anthropic"

    def test_load_missing_file_raises_error(self) -> None:
        """Verify ConfigurationError is raised for missing files."""
        # Attempt to load a non-existent file
        with pytest.raises(ConfigurationError, match="not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        """Verify ConfigurationError is raised for invalid YAML."""
        # Write invalid YAML to a temp file
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid: yaml: [}", encoding="utf-8")
        # Attempt to load should raise
        with pytest.raises(ConfigurationError, match="parse"):
            load_config(config_file)

    def test_load_empty_config_uses_defaults(self, tmp_path: Path) -> None:
        """Verify an empty YAML file returns default config."""
        # Write an empty file
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")
        # Load should return defaults
        config = load_config(config_file)
        assert config.environment == "dev"

    def test_load_partial_config_fills_defaults(self, tmp_path: Path) -> None:
        """Verify partial config is merged with defaults."""
        # Write config with only some fields
        config_data = {"environment": "staging"}
        config_file = tmp_path / "partial.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        # Load and verify defaults fill in
        config = load_config(config_file)
        assert config.environment == "staging"
        assert config.provider.name == "openai"


class TestLoadConfigFromEnv:
    """Tests for the load_config_from_env function."""

    def test_uses_env_variable_path(self, tmp_path: Path) -> None:
        """Verify the function reads from CONTEXT_MANAGER_CONFIG_PATH."""
        # Write a valid config file
        config_data = {"environment": "from_env"}
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        # Set the environment variable
        os.environ["CONTEXT_MANAGER_CONFIG_PATH"] = str(config_file)
        try:
            # Load using the env variable
            config = load_config_from_env()
            assert config.environment == "from_env"
        finally:
            # Clean up the environment variable
            del os.environ["CONTEXT_MANAGER_CONFIG_PATH"]

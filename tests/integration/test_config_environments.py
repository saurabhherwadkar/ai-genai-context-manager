"""Integration tests for multi-environment configuration loading."""

from pathlib import Path

import pytest

from context_manager.config import load_config


class TestConfigEnvironments:
    """Tests verifying all environment config files load correctly."""

    @pytest.fixture
    def config_dir(self) -> Path:
        """Get the path to the config directory."""
        # Navigate from tests to the project root config directory
        return Path(__file__).parent.parent.parent / "config"

    def test_dev_config_loads(self, config_dir: Path) -> None:
        """Verify the dev config file loads and validates."""
        # Load the dev configuration
        config = load_config(config_dir / "config.dev.yaml")
        # Verify environment is set correctly
        assert config.environment == "dev"
        # Verify provider settings
        assert config.provider.name == "openai"
        # Verify logging level for dev
        assert config.logging.level == "DEBUG"

    def test_staging_config_loads(self, config_dir: Path) -> None:
        """Verify the staging config file loads and validates."""
        # Load the staging configuration
        config = load_config(config_dir / "config.staging.yaml")
        # Verify environment is set correctly
        assert config.environment == "staging"
        # Verify logging uses structured format
        assert config.logging.format == "structured"

    def test_prod_config_loads(self, config_dir: Path) -> None:
        """Verify the prod config file loads and validates."""
        # Load the prod configuration
        config = load_config(config_dir / "config.prod.yaml")
        # Verify environment is set correctly
        assert config.environment == "prod"
        # Verify prod has stricter security limits
        assert config.security.max_message_length <= 100000
        # Verify prod logging is less verbose
        assert config.logging.level == "WARNING"

    def test_all_configs_have_valid_provider(self, config_dir: Path) -> None:
        """Verify all configs specify a valid provider name."""
        # Load all environment configs
        for env in ["dev", "staging", "prod"]:
            config = load_config(config_dir / f"config.{env}.yaml")
            # Provider must be openai or anthropic
            assert config.provider.name in ("openai", "anthropic")

    def test_all_configs_have_valid_trimming_strategy(self, config_dir: Path) -> None:
        """Verify all configs specify a valid trimming strategy."""
        # Load all environment configs
        for env in ["dev", "staging", "prod"]:
            config = load_config(config_dir / f"config.{env}.yaml")
            # Strategy must be one of the supported options
            assert config.trimming.strategy in ("fifo", "sliding_window", "priority")

"""Configuration loading and validation using Pydantic models.

Loads YAML configuration files and validates them against
strongly-typed Pydantic models. Supports environment-specific
config files (dev, staging, prod).
"""

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from context_manager.exceptions import ConfigurationError

# Module-level logger for config loading diagnostics
logger = logging.getLogger(__name__)


class ProviderConfig(BaseModel, frozen=True):
    """Configuration for the LLM provider connection settings."""

    # Which LLM provider to use for primary operations
    name: Literal["openai", "anthropic"] = "openai"
    # Model identifier for completions and summarization
    model: str = "gpt-4o"
    # Maximum context window size in tokens for the chosen model
    max_context_tokens: int = 128_000
    # Tokens reserved for the model's response generation
    reserved_response_tokens: int = 4096
    # HTTP request timeout in seconds for provider API calls
    timeout_seconds: int = 30
    # Maximum number of retries on transient failures
    max_retries: int = 3

    @field_validator("max_context_tokens")
    @classmethod
    def validate_max_context_tokens(cls, value: int) -> int:
        """Ensure the context window size is a positive integer."""
        # Context window must be at least 1 token to be valid
        if value <= 0:
            raise ValueError("max_context_tokens must be positive")
        # Return the validated value unchanged
        return value

    @field_validator("reserved_response_tokens")
    @classmethod
    def validate_reserved_response_tokens(cls, value: int) -> int:
        """Ensure the reserved response tokens is non-negative."""
        # Reserved tokens cannot be negative
        if value < 0:
            raise ValueError("reserved_response_tokens must be non-negative")
        # Return the validated value unchanged
        return value


class TokenCountingConfig(BaseModel, frozen=True):
    """Configuration for the token counting strategy."""

    # Which counting strategy to use: tiktoken (local), api, or estimator
    strategy: Literal["tiktoken", "api", "estimator"] = "tiktoken"
    # Tiktoken encoding name for OpenAI-compatible token counting
    encoding: str = "cl100k_base"
    # Characters-per-token ratio used by the estimator fallback
    estimator_ratio: float = 4.0

    @field_validator("estimator_ratio")
    @classmethod
    def validate_estimator_ratio(cls, value: float) -> float:
        """Ensure the estimator ratio is a positive number."""
        # Ratio must be positive to produce valid token estimates
        if value <= 0:
            raise ValueError("estimator_ratio must be positive")
        # Return the validated ratio value
        return value


class TrimmingConfig(BaseModel, frozen=True):
    """Configuration for the message trimming strategy."""

    # Which trimming strategy to apply: fifo, sliding_window, or priority
    strategy: Literal["fifo", "sliding_window", "priority"] = "fifo"
    # Number of recent messages to retain in sliding window mode
    window_size: int = 50
    # Whether to always preserve the system message from trimming
    preserve_system_message: bool = True

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, value: int) -> int:
        """Ensure the sliding window size is positive."""
        # Window size must allow at least one message
        if value <= 0:
            raise ValueError("window_size must be positive")
        # Return the validated window size
        return value


class SummarizationConfig(BaseModel, frozen=True):
    """Configuration for the optional summarization feature."""

    # Whether summarization is enabled (false = pure trimming only)
    enabled: bool = False
    # Which provider to use for summarization API calls
    provider: Literal["openai", "anthropic"] = "openai"
    # Model to use for generating summaries
    model: str = "gpt-4o-mini"
    # Maximum tokens allowed for the generated summary
    max_summary_tokens: int = 500
    # Template for the summarization prompt with {messages} placeholder
    prompt_template: str = (
        "Summarize the following conversation concisely, "
        "preserving key context and decisions:\n\n{messages}"
    )

    @field_validator("max_summary_tokens")
    @classmethod
    def validate_max_summary_tokens(cls, value: int) -> int:
        """Ensure the summary token limit is positive."""
        # Summary must allow at least 1 token of output
        if value <= 0:
            raise ValueError("max_summary_tokens must be positive")
        # Return the validated token limit
        return value


class LoggingConfig(BaseModel, frozen=True):
    """Configuration for the structured logging system."""

    # Log level threshold: DEBUG, INFO, WARNING, ERROR, or CRITICAL
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # Output format: structured (JSON lines) or human (readable text)
    format: Literal["structured", "human"] = "human"
    # Optional file path for log output (None = stderr only)
    file: str | None = None


class SecurityConfig(BaseModel, frozen=True):
    """Configuration for input validation and security controls."""

    # Maximum allowed character length for any single message
    max_message_length: int = 100_000
    # Maximum number of messages allowed in a conversation
    max_messages: int = 1000
    # Whether to strip control characters from message content
    sanitize_input: bool = True

    @field_validator("max_message_length")
    @classmethod
    def validate_max_message_length(cls, value: int) -> int:
        """Ensure the message length limit is positive."""
        # Must allow at least one character per message
        if value <= 0:
            raise ValueError("max_message_length must be positive")
        # Return the validated length limit
        return value

    @field_validator("max_messages")
    @classmethod
    def validate_max_messages(cls, value: int) -> int:
        """Ensure the message count limit is positive."""
        # Must allow at least one message in a conversation
        if value <= 0:
            raise ValueError("max_messages must be positive")
        # Return the validated count limit
        return value


class ContextManagerConfig(BaseModel, frozen=True):
    """Root configuration model containing all subsections.

    This is the top-level config object loaded from YAML files.
    All fields have sensible defaults for development use.
    """

    # Environment identifier for logging and diagnostic purposes
    environment: str = "dev"
    # LLM provider connection configuration
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    # Token counting strategy configuration
    token_counting: TokenCountingConfig = Field(default_factory=TokenCountingConfig)
    # Message trimming strategy configuration
    trimming: TrimmingConfig = Field(default_factory=TrimmingConfig)
    # Optional summarization feature configuration
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    # Structured logging configuration
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Input validation and security configuration
    security: SecurityConfig = Field(default_factory=SecurityConfig)


def load_config(config_path: str | Path) -> ContextManagerConfig:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A validated ContextManagerConfig instance.

    Raises:
        ConfigurationError: If the file cannot be read or parsed.
    """
    # Convert string paths to Path objects for consistent handling
    path = Path(config_path)
    # Verify the configuration file exists before attempting to read
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    # Attempt to read and parse the YAML file contents
    try:
        # Open the file with UTF-8 encoding for cross-platform compatibility
        with path.open(encoding="utf-8") as file:
            # Parse the YAML content into a Python dictionary
            raw_config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        # Wrap YAML parsing errors in our custom exception type
        raise ConfigurationError(f"Failed to parse YAML config: {exc}") from exc
    except OSError as exc:
        # Wrap file system errors in our custom exception type
        raise ConfigurationError(f"Failed to read config file: {exc}") from exc
    # Handle the case where the YAML file is empty
    if raw_config is None:
        # Log a warning and return defaults for empty config files
        logger.warning("Config file is empty, using defaults: %s", path)
        # Return a config instance with all default values
        return ContextManagerConfig()
    # Validate the raw dictionary against the Pydantic model
    try:
        # Pydantic validates all fields and nested models recursively
        config = ContextManagerConfig(**raw_config)
    except Exception as exc:
        # Wrap Pydantic validation errors in our custom exception type
        raise ConfigurationError(f"Configuration validation failed: {exc}") from exc
    # Log successful configuration loading at debug level
    logger.debug("Configuration loaded successfully from: %s", path)
    # Return the fully validated configuration object
    return config


def load_config_from_env() -> ContextManagerConfig:
    """Load configuration from the path specified in environment variable.

    Reads the CONTEXT_MANAGER_CONFIG_PATH environment variable to find
    the configuration file. Falls back to config/config.dev.yaml if unset.

    Returns:
        A validated ContextManagerConfig instance.

    Raises:
        ConfigurationError: If the resolved path cannot be loaded.
    """
    # Read the config path from the environment variable
    config_path = os.environ.get("CONTEXT_MANAGER_CONFIG_PATH", "config/config.dev.yaml")
    # Log which configuration file path was resolved
    logger.info("Loading configuration from: %s", config_path)
    # Delegate to the path-based loader for actual loading
    return load_config(config_path)

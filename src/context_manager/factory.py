"""Factory functions for creating ContextManager instances from configuration.

Provides convenience functions that read configuration and instantiate
the appropriate strategies based on config values, wiring everything
together without manual dependency injection.
"""

import logging
from pathlib import Path

from context_manager.config import ContextManagerConfig, load_config, load_config_from_env
from context_manager.logging_setup import setup_logging
from context_manager.manager import ContextManager
from context_manager.protocols import SummarizationStrategy, TokenCounter, TrimmingStrategy
from context_manager.summarization.anthropic_summarizer import AnthropicSummarizer
from context_manager.summarization.openai_summarizer import OpenAISummarizer
from context_manager.token_counting.anthropic_counter import AnthropicTokenCounter
from context_manager.token_counting.estimator import EstimatorTokenCounter
from context_manager.token_counting.openai_counter import TiktokenCounter
from context_manager.trimming.fifo import FifoTrimmingStrategy
from context_manager.trimming.priority import PriorityTrimmingStrategy
from context_manager.trimming.sliding_window import SlidingWindowStrategy

# Module-level logger for factory diagnostics
logger = logging.getLogger(__name__)


def create_from_config(config_path: str | Path) -> ContextManager:
    """Create a fully configured ContextManager from a YAML config file.

    Loads the configuration, sets up logging, and instantiates all
    strategies based on the config values.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A fully configured ContextManager instance ready for use.
    """
    # Load and validate the configuration from the YAML file
    config = load_config(config_path)
    # Configure logging based on the loaded settings
    setup_logging(config.logging)
    # Build and return the context manager with resolved strategies
    return _build_manager(config)


def create_from_env() -> ContextManager:
    """Create a ContextManager using the environment-specified config.

    Reads the config path from the CONTEXT_MANAGER_CONFIG_PATH env var,
    falling back to config/config.dev.yaml if not set.

    Returns:
        A fully configured ContextManager instance ready for use.
    """
    # Load configuration from the environment-specified path
    config = load_config_from_env()
    # Configure logging based on the loaded settings
    setup_logging(config.logging)
    # Build and return the context manager with resolved strategies
    return _build_manager(config)


def _build_manager(config: ContextManagerConfig) -> ContextManager:
    """Build a ContextManager with strategies resolved from configuration.

    Internal factory that maps config string values to concrete
    strategy implementations.

    Args:
        config: The validated configuration to build from.

    Returns:
        A configured ContextManager instance.
    """
    # Resolve the token counter strategy from config
    token_counter = _resolve_token_counter(config)
    # Resolve the trimming strategy from config
    trimming_strategy = _resolve_trimming_strategy(config)
    # Resolve the optional summarization strategy from config
    summarization_strategy = _resolve_summarization_strategy(config)
    # Log the resolved strategy types
    logger.info(
        "Building ContextManager (counter=%s, trimmer=%s, summarizer=%s)",
        type(token_counter).__name__,
        type(trimming_strategy).__name__,
        type(summarization_strategy).__name__ if summarization_strategy else "None",
    )
    # Create and return the wired ContextManager instance
    return ContextManager(
        config=config,
        token_counter=token_counter,
        trimming_strategy=trimming_strategy,
        summarization_strategy=summarization_strategy,
    )


def _resolve_token_counter(config: ContextManagerConfig) -> TokenCounter:
    """Resolve the token counter implementation from configuration.

    Maps the strategy string to the appropriate counter class.

    Args:
        config: Configuration with token counting settings.

    Returns:
        An initialized TokenCounter implementation.
    """
    # Select the counter based on the configured strategy name
    match config.token_counting.strategy:
        case "tiktoken":
            # Use the local tiktoken-based counter
            return TiktokenCounter(encoding=config.token_counting.encoding)
        case "api":
            # Use the Anthropic API-based counter
            return AnthropicTokenCounter(
                model=config.provider.model,
                fallback_ratio=config.token_counting.estimator_ratio,
            )
        case "estimator":
            # Use the character-ratio estimator fallback
            return EstimatorTokenCounter(ratio=config.token_counting.estimator_ratio)
        case _:
            # Default to tiktoken for unrecognized strategy names
            logger.warning(
                "Unknown token counting strategy '%s', defaulting to tiktoken",
                config.token_counting.strategy,
            )
            return TiktokenCounter(encoding=config.token_counting.encoding)


def _resolve_trimming_strategy(config: ContextManagerConfig) -> TrimmingStrategy:
    """Resolve the trimming strategy implementation from configuration.

    Maps the strategy string to the appropriate trimmer class.

    Args:
        config: Configuration with trimming settings.

    Returns:
        An initialized TrimmingStrategy implementation.
    """
    # Select the trimmer based on the configured strategy name
    match config.trimming.strategy:
        case "fifo":
            # Use the first-in-first-out trimming strategy
            return FifoTrimmingStrategy(
                preserve_system_message=config.trimming.preserve_system_message,
            )
        case "sliding_window":
            # Use the sliding window trimming strategy
            return SlidingWindowStrategy(
                window_size=config.trimming.window_size,
                preserve_system_message=config.trimming.preserve_system_message,
            )
        case "priority":
            # Use the priority-based trimming strategy
            return PriorityTrimmingStrategy(
                preserve_system_message=config.trimming.preserve_system_message,
            )
        case _:
            # Default to FIFO for unrecognized strategy names
            logger.warning(
                "Unknown trimming strategy '%s', defaulting to FIFO",
                config.trimming.strategy,
            )
            return FifoTrimmingStrategy(
                preserve_system_message=config.trimming.preserve_system_message,
            )


def _resolve_summarization_strategy(
    config: ContextManagerConfig,
) -> SummarizationStrategy | None:
    """Resolve the summarization strategy from configuration.

    Returns None if summarization is disabled in the config.

    Args:
        config: Configuration with summarization settings.

    Returns:
        An initialized SummarizationStrategy or None if disabled.
    """
    # Return None if summarization is not enabled
    if not config.summarization.enabled:
        return None
    # Select the summarizer based on the configured provider name
    match config.summarization.provider:
        case "openai":
            # Use the OpenAI-based summarizer
            return OpenAISummarizer(
                model=config.summarization.model,
                max_summary_tokens=config.summarization.max_summary_tokens,
                prompt_template=config.summarization.prompt_template,
            )
        case "anthropic":
            # Use the Anthropic-based summarizer
            return AnthropicSummarizer(
                model=config.summarization.model,
                max_summary_tokens=config.summarization.max_summary_tokens,
                prompt_template=config.summarization.prompt_template,
            )
        case _:
            # Default to OpenAI for unrecognized provider names
            logger.warning(
                "Unknown summarization provider '%s', defaulting to OpenAI",
                config.summarization.provider,
            )
            return OpenAISummarizer(
                model=config.summarization.model,
                max_summary_tokens=config.summarization.max_summary_tokens,
                prompt_template=config.summarization.prompt_template,
            )

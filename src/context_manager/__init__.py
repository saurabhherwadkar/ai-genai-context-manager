"""AI GenAI Context Manager - Framework-agnostic LLM context window management.

This package provides pluggable strategies for trimming, summarizing,
and counting tokens in LLM conversation histories. Supports both
OpenAI and Anthropic as configurable providers.

Basic usage:
    from context_manager import create_from_config, Message, Role

    manager = create_from_config("config/config.dev.yaml")
    await manager.add_message(Message(role=Role.USER, content="Hello!"))
    context = await manager.get_context_window()
"""

from context_manager._version import __version__
from context_manager.config import ContextManagerConfig, load_config, load_config_from_env
from context_manager.exceptions import (
    ConfigurationError,
    ContextManagerError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SummarizationError,
    TokenBudgetExceededError,
    TokenCountError,
    TrimmingError,
    ValidationError,
)
from context_manager.factory import create_from_config, create_from_env
from context_manager.manager import ContextManager
from context_manager.models import ConversationState, Message, Priority, Role, TokenCount
from context_manager.protocols import (
    LLMProvider,
    SummarizationStrategy,
    TokenCounter,
    TrimmingStrategy,
)

# Public API exports for star imports
__all__ = [
    # Version
    "__version__",
    # Main classes
    "ContextManager",
    "ContextManagerConfig",
    # Factory functions
    "create_from_config",
    "create_from_env",
    "load_config",
    "load_config_from_env",
    # Data models
    "ConversationState",
    "Message",
    "Priority",
    "Role",
    "TokenCount",
    # Protocols
    "LLMProvider",
    "SummarizationStrategy",
    "TokenCounter",
    "TrimmingStrategy",
    # Exceptions
    "ConfigurationError",
    "ContextManagerError",
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "SummarizationError",
    "TokenBudgetExceededError",
    "TokenCountError",
    "TrimmingError",
    "ValidationError",
]

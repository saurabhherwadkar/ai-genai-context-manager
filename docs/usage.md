# Usage Guide

## Overview

The `context_manager` package provides tools to manage LLM conversation context windows. It automatically handles token counting, trimming excess messages, and optionally summarizing older conversation segments.

## Core Concepts

### Messages

Messages are immutable Pydantic models with role, content, and priority:

```python
from context_manager import Message, Role, Priority

# System messages guide the assistant (typically critical priority)
system_msg = Message(
    role=Role.SYSTEM,
    content="You are a helpful assistant.",
    priority=Priority.CRITICAL,
)

# User messages are standard priority by default
user_msg = Message(role=Role.USER, content="Hello!")

# You can set priority to control trimming order
important_msg = Message(
    role=Role.USER,
    content="Important context",
    priority=Priority.HIGH,
)
```

### Priority Levels

- **CRITICAL**: Never trimmed (use for system prompts)
- **HIGH**: Trimmed last
- **NORMAL**: Default trimming priority
- **LOW**: Trimmed first

### Token Counting

Three strategies are available:

1. **tiktoken** (default): Fast, local counting for OpenAI models
2. **api**: Uses Anthropic's count_tokens endpoint
3. **estimator**: Character-ratio fallback (no dependencies)

### Trimming Strategies

1. **fifo**: Removes oldest messages first
2. **sliding_window**: Keeps only the N most recent messages
3. **priority**: Removes lowest-priority messages first

### Summarization

When enabled, older messages are summarized into a single condensed message before trimming. This preserves semantic context while reducing token usage.

## Configuration

All behavior is controlled via YAML configuration files. Set the `CONTEXT_MANAGER_CONFIG_PATH` environment variable to select which file to load.

See `config/config.dev.yaml` for the full reference with all options documented.

## Error Handling

The package provides a structured exception hierarchy:

```python
from context_manager import (
    ContextManagerError,       # Base: catch all package errors
    ConfigurationError,        # Invalid config file
    TokenBudgetExceededError,  # Single message too large
    TrimmingError,             # Cannot trim enough
    SummarizationError,       # Summarization API failed
    ProviderError,             # LLM API call failed
    ValidationError,           # Input validation failed
)
```

## Extending

Implement the Protocol interfaces to add custom strategies:

```python
from collections.abc import Sequence
from context_manager.models import Message, TokenCount

class MyCustomCounter:
    """Custom token counter - just implement the protocol methods."""

    async def count_tokens(self, messages: Sequence[Message]) -> TokenCount:
        # Your counting logic here
        ...

    async def count_single(self, text: str) -> int:
        # Your single-text counting logic
        ...
```

No inheritance required - any object matching the Protocol signature works.

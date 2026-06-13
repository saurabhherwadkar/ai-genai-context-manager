"""Shared test fixtures for the context_manager test suite.

Provides reusable fixtures for configuration, mock clients,
sample messages, and pre-configured strategy instances.
"""

import pytest

from context_manager.config import ContextManagerConfig
from context_manager.models import Message, Priority, Role


@pytest.fixture
def dev_config() -> ContextManagerConfig:
    """Create a development configuration for testing."""
    # Return a default config suitable for unit tests
    return ContextManagerConfig(
        environment="test",
    )


@pytest.fixture
def small_budget_config() -> ContextManagerConfig:
    """Create a config with a very small token budget for trim testing."""
    # Return a config with tight token limits to trigger trimming
    return ContextManagerConfig(
        environment="test",
        provider={  # type: ignore[arg-type]
            "name": "openai",
            "model": "gpt-4o",
            "max_context_tokens": 100,
            "reserved_response_tokens": 20,
        },
        trimming={  # type: ignore[arg-type]
            "strategy": "fifo",
            "preserve_system_message": True,
            "window_size": 5,
        },
        summarization={"enabled": False},  # type: ignore[arg-type]
    )


@pytest.fixture
def sample_system_message() -> Message:
    """Create a sample system message for testing."""
    # Return a system message with critical priority
    return Message(
        role=Role.SYSTEM,
        content="You are a helpful assistant.",
        priority=Priority.CRITICAL,
    )


@pytest.fixture
def sample_user_message() -> Message:
    """Create a sample user message for testing."""
    # Return a standard user message
    return Message(
        role=Role.USER,
        content="What is Python?",
        priority=Priority.NORMAL,
    )


@pytest.fixture
def sample_assistant_message() -> Message:
    """Create a sample assistant message for testing."""
    # Return a standard assistant response message
    return Message(
        role=Role.ASSISTANT,
        content="Python is a high-level programming language.",
        priority=Priority.NORMAL,
    )


@pytest.fixture
def sample_messages(
    sample_system_message: Message,
    sample_user_message: Message,
    sample_assistant_message: Message,
) -> list[Message]:
    """Create a list of sample messages for testing."""
    # Return a typical conversation sequence
    return [sample_system_message, sample_user_message, sample_assistant_message]


@pytest.fixture
def many_messages() -> list[Message]:
    """Create a large list of messages for trimming tests."""
    # Start with a system message
    messages = [
        Message(role=Role.SYSTEM, content="You are helpful.", priority=Priority.CRITICAL)
    ]
    # Add 20 user/assistant pairs to create a long conversation
    for i in range(20):
        messages.append(
            Message(role=Role.USER, content=f"Question number {i}: What is item {i}?")
        )
        messages.append(
            Message(role=Role.ASSISTANT, content=f"Answer number {i}: Item {i} is a thing.")
        )
    # Return the complete conversation with 41 messages
    return messages


@pytest.fixture
def priority_messages() -> list[Message]:
    """Create messages with various priority levels for priority trimming tests."""
    # Return messages spanning all priority levels
    return [
        Message(role=Role.SYSTEM, content="System prompt.", priority=Priority.CRITICAL),
        Message(role=Role.USER, content="Low priority msg.", priority=Priority.LOW),
        Message(role=Role.ASSISTANT, content="Normal reply.", priority=Priority.NORMAL),
        Message(role=Role.USER, content="High priority msg.", priority=Priority.HIGH),
        Message(role=Role.ASSISTANT, content="Another normal.", priority=Priority.NORMAL),
        Message(role=Role.USER, content="Another low.", priority=Priority.LOW),
        Message(role=Role.ASSISTANT, content="Critical info.", priority=Priority.CRITICAL),
    ]

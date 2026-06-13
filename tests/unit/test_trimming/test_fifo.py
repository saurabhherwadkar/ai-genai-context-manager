"""Unit tests for the FIFO trimming strategy."""

import pytest

from context_manager.models import Message, Priority, Role
from context_manager.trimming.fifo import FifoTrimmingStrategy


class TestFifoTrimmingStrategy:
    """Tests for the FifoTrimmingStrategy implementation."""

    @pytest.fixture
    def strategy(self) -> FifoTrimmingStrategy:
        """Create a FIFO strategy with system message preservation."""
        # Initialize with default settings
        return FifoTrimmingStrategy(preserve_system_message=True)

    def test_no_trimming_when_within_budget(self, strategy: FifoTrimmingStrategy) -> None:
        """Verify no messages are removed when within the token budget."""
        # Create messages with known token counts
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi"),
        ]
        token_counts = [5, 3]
        # Trim with a budget larger than total
        result = strategy.trim(messages, max_tokens=100, token_counts=token_counts)
        # All messages should be retained
        assert len(result) == 2

    def test_trims_oldest_first(self, strategy: FifoTrimmingStrategy) -> None:
        """Verify the oldest messages are trimmed first."""
        # Create messages in chronological order
        messages = [
            Message(role=Role.USER, content="First"),
            Message(role=Role.USER, content="Second"),
            Message(role=Role.USER, content="Third"),
        ]
        token_counts = [10, 10, 10]
        # Trim to fit only 20 tokens (need to remove 1 message)
        result = strategy.trim(messages, max_tokens=20, token_counts=token_counts)
        # Should have removed the oldest message
        assert len(result) == 2
        assert result[0].content == "Second"
        assert result[1].content == "Third"

    def test_preserves_system_message(self, strategy: FifoTrimmingStrategy) -> None:
        """Verify system messages are never trimmed when preservation is on."""
        # Create messages with a system message first
        messages = [
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="User msg"),
            Message(role=Role.ASSISTANT, content="Response"),
        ]
        token_counts = [10, 10, 10]
        # Trim to 15 tokens (must remove 2 non-system messages)
        result = strategy.trim(messages, max_tokens=15, token_counts=token_counts)
        # System message should remain
        assert result[0].role == Role.SYSTEM

    def test_preserves_critical_priority(self, strategy: FifoTrimmingStrategy) -> None:
        """Verify critical priority messages are never trimmed."""
        # Create messages with critical priority
        messages = [
            Message(role=Role.USER, content="Critical", priority=Priority.CRITICAL),
            Message(role=Role.USER, content="Normal", priority=Priority.NORMAL),
            Message(role=Role.USER, content="Also normal", priority=Priority.NORMAL),
        ]
        token_counts = [10, 10, 10]
        # Trim to 15 tokens
        result = strategy.trim(messages, max_tokens=15, token_counts=token_counts)
        # Critical message should remain
        assert any(msg.content == "Critical" for msg in result)

    def test_no_system_preservation(self) -> None:
        """Verify system messages are trimmed when preservation is off."""
        # Create strategy without system message preservation
        strategy = FifoTrimmingStrategy(preserve_system_message=False)
        # Create messages with system message
        messages = [
            Message(role=Role.SYSTEM, content="System"),
            Message(role=Role.USER, content="User"),
        ]
        token_counts = [10, 10]
        # Trim to 10 tokens - system message is oldest and should go
        result = strategy.trim(messages, max_tokens=10, token_counts=token_counts)
        # System message should be removed (it's oldest)
        assert len(result) == 1
        assert result[0].content == "User"

    def test_empty_message_list(self, strategy: FifoTrimmingStrategy) -> None:
        """Verify empty input returns empty output."""
        # Trim with no messages
        result = strategy.trim([], max_tokens=100, token_counts=[])
        # Should return empty list
        assert result == []

    def test_returns_new_list_instance(self, strategy: FifoTrimmingStrategy) -> None:
        """Verify the result is a new list, not a reference to the input."""
        # Create messages within budget
        messages = [Message(role=Role.USER, content="Test")]
        token_counts = [5]
        # Trim (no actual trimming needed)
        result = strategy.trim(messages, max_tokens=100, token_counts=token_counts)
        # Should be a different list object
        assert result is not messages

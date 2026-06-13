"""Unit tests for the priority-based trimming strategy."""

import pytest

from context_manager.models import Message, Priority, Role
from context_manager.trimming.priority import PriorityTrimmingStrategy


class TestPriorityTrimmingStrategy:
    """Tests for the PriorityTrimmingStrategy implementation."""

    @pytest.fixture
    def strategy(self) -> PriorityTrimmingStrategy:
        """Create a priority strategy with system message preservation."""
        # Initialize with default settings
        return PriorityTrimmingStrategy(preserve_system_message=True)

    def test_no_trimming_when_within_budget(self, strategy: PriorityTrimmingStrategy) -> None:
        """Verify no messages are removed when within the token budget."""
        # Create messages within budget
        messages = [
            Message(role=Role.USER, content="Hello", priority=Priority.NORMAL),
        ]
        token_counts = [5]
        # Trim with generous budget
        result = strategy.trim(messages, max_tokens=100, token_counts=token_counts)
        # All messages should remain
        assert len(result) == 1

    def test_removes_low_priority_first(self, strategy: PriorityTrimmingStrategy) -> None:
        """Verify LOW priority messages are removed before NORMAL."""
        # Create messages with mixed priorities
        messages = [
            Message(role=Role.USER, content="Normal msg", priority=Priority.NORMAL),
            Message(role=Role.USER, content="Low msg", priority=Priority.LOW),
            Message(role=Role.USER, content="High msg", priority=Priority.HIGH),
        ]
        token_counts = [10, 10, 10]
        # Budget allows only 2 messages
        result = strategy.trim(messages, max_tokens=20, token_counts=token_counts)
        # Low priority should be removed first
        assert not any(msg.content == "Low msg" for msg in result)
        # High priority should remain
        assert any(msg.content == "High msg" for msg in result)

    def test_removes_normal_before_high(self, strategy: PriorityTrimmingStrategy) -> None:
        """Verify NORMAL priority messages are removed before HIGH."""
        # Create messages where we need to remove normal to fit
        messages = [
            Message(role=Role.USER, content="High", priority=Priority.HIGH),
            Message(role=Role.USER, content="Normal", priority=Priority.NORMAL),
        ]
        token_counts = [10, 10]
        # Budget allows only 1 message
        result = strategy.trim(messages, max_tokens=10, token_counts=token_counts)
        # Normal should be removed, high should remain
        assert len(result) == 1
        assert result[0].content == "High"

    def test_never_removes_critical(self, strategy: PriorityTrimmingStrategy) -> None:
        """Verify CRITICAL priority messages are never removed."""
        # Create messages with critical priority
        messages = [
            Message(role=Role.USER, content="Critical", priority=Priority.CRITICAL),
            Message(role=Role.USER, content="Low", priority=Priority.LOW),
        ]
        token_counts = [10, 10]
        # Budget that requires trimming
        result = strategy.trim(messages, max_tokens=15, token_counts=token_counts)
        # Critical should remain
        assert any(msg.content == "Critical" for msg in result)
        # Low should be removed
        assert not any(msg.content == "Low" for msg in result)

    def test_preserves_system_messages(self, strategy: PriorityTrimmingStrategy) -> None:
        """Verify system messages are preserved when configured."""
        # Create with system message
        messages = [
            Message(role=Role.SYSTEM, content="System", priority=Priority.NORMAL),
            Message(role=Role.USER, content="User", priority=Priority.NORMAL),
        ]
        token_counts = [10, 10]
        # Budget allows only 1 message
        result = strategy.trim(messages, max_tokens=10, token_counts=token_counts)
        # System message should remain even with NORMAL priority
        assert any(msg.role == Role.SYSTEM for msg in result)

    def test_priority_order_within_same_level(
        self, strategy: PriorityTrimmingStrategy
    ) -> None:
        """Verify messages at the same priority level are trimmed oldest first."""
        # Create multiple messages at the same priority
        messages = [
            Message(role=Role.USER, content="First normal", priority=Priority.NORMAL),
            Message(role=Role.USER, content="Second normal", priority=Priority.NORMAL),
            Message(role=Role.USER, content="Third normal", priority=Priority.NORMAL),
        ]
        token_counts = [10, 10, 10]
        # Budget allows 2 messages
        result = strategy.trim(messages, max_tokens=20, token_counts=token_counts)
        # Should remove the first (oldest) normal message
        assert len(result) == 2
        assert result[0].content == "Second normal"
        assert result[1].content == "Third normal"

    def test_empty_message_list(self, strategy: PriorityTrimmingStrategy) -> None:
        """Verify empty input returns empty output."""
        # Trim with no messages
        result = strategy.trim([], max_tokens=100, token_counts=[])
        # Should return empty list
        assert result == []

    def test_full_priority_order(self, strategy: PriorityTrimmingStrategy) -> None:
        """Verify the complete priority trimming order."""
        # Create one message of each priority level
        messages = [
            Message(role=Role.USER, content="Low", priority=Priority.LOW),
            Message(role=Role.USER, content="Normal", priority=Priority.NORMAL),
            Message(role=Role.USER, content="High", priority=Priority.HIGH),
            Message(role=Role.USER, content="Critical", priority=Priority.CRITICAL),
        ]
        token_counts = [10, 10, 10, 10]
        # Budget for only 1 message - should keep critical
        result = strategy.trim(messages, max_tokens=10, token_counts=token_counts)
        # Only critical should remain
        assert len(result) == 1
        assert result[0].priority == Priority.CRITICAL

"""Unit tests for the sliding window trimming strategy."""

import pytest

from context_manager.models import Message, Role
from context_manager.trimming.sliding_window import SlidingWindowStrategy


class TestSlidingWindowStrategy:
    """Tests for the SlidingWindowStrategy implementation."""

    @pytest.fixture
    def strategy(self) -> SlidingWindowStrategy:
        """Create a sliding window strategy with window size 3."""
        # Use a small window for easy testing
        return SlidingWindowStrategy(window_size=3, preserve_system_message=True)

    def test_no_trimming_when_within_window(self, strategy: SlidingWindowStrategy) -> None:
        """Verify no trimming when messages fit within the window."""
        # Create fewer messages than the window size
        messages = [
            Message(role=Role.USER, content="One"),
            Message(role=Role.ASSISTANT, content="Two"),
        ]
        token_counts = [5, 5]
        # Trim with large token budget
        result = strategy.trim(messages, max_tokens=100, token_counts=token_counts)
        # All messages should be retained
        assert len(result) == 2

    def test_keeps_most_recent_messages(self, strategy: SlidingWindowStrategy) -> None:
        """Verify only the most recent messages within the window are kept."""
        # Create more messages than the window allows
        messages = [
            Message(role=Role.USER, content="First"),
            Message(role=Role.USER, content="Second"),
            Message(role=Role.USER, content="Third"),
            Message(role=Role.USER, content="Fourth"),
            Message(role=Role.USER, content="Fifth"),
        ]
        token_counts = [5, 5, 5, 5, 5]
        # Trim to the window (should keep last 3)
        result = strategy.trim(messages, max_tokens=100, token_counts=token_counts)
        # Should keep the 3 most recent messages
        assert len(result) == 3
        assert result[0].content == "Third"
        assert result[1].content == "Fourth"
        assert result[2].content == "Fifth"

    def test_preserves_system_message_outside_window(
        self, strategy: SlidingWindowStrategy
    ) -> None:
        """Verify system messages are kept even if outside the window."""
        # Create messages with system message at the start
        messages = [
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="First"),
            Message(role=Role.USER, content="Second"),
            Message(role=Role.USER, content="Third"),
            Message(role=Role.USER, content="Fourth"),
        ]
        token_counts = [5, 5, 5, 5, 5]
        # Window is 3, system takes 1 slot, so 2 recent messages fit
        result = strategy.trim(messages, max_tokens=100, token_counts=token_counts)
        # System message should be preserved
        assert result[0].role == Role.SYSTEM
        # Most recent messages should be included
        assert result[-1].content == "Fourth"

    def test_applies_token_budget_after_windowing(self) -> None:
        """Verify token budget is enforced after window selection."""
        # Create a strategy with a large window
        strategy = SlidingWindowStrategy(window_size=10, preserve_system_message=True)
        # Create messages that fit in window but exceed token budget
        messages = [
            Message(role=Role.USER, content="A"),
            Message(role=Role.USER, content="B"),
            Message(role=Role.USER, content="C"),
        ]
        token_counts = [30, 30, 30]
        # Token budget is less than all messages combined
        result = strategy.trim(messages, max_tokens=50, token_counts=token_counts)
        # Should trim to fit the token budget
        assert len(result) < 3

    def test_empty_message_list(self, strategy: SlidingWindowStrategy) -> None:
        """Verify empty input returns empty output."""
        # Trim with no messages
        result = strategy.trim([], max_tokens=100, token_counts=[])
        # Should return empty list
        assert result == []

    def test_window_size_one(self) -> None:
        """Verify window size of 1 keeps only the most recent message."""
        # Create strategy with minimum window
        strategy = SlidingWindowStrategy(window_size=1, preserve_system_message=False)
        # Create multiple messages
        messages = [
            Message(role=Role.USER, content="Old"),
            Message(role=Role.USER, content="New"),
        ]
        token_counts = [5, 5]
        # Should keep only the newest
        result = strategy.trim(messages, max_tokens=100, token_counts=token_counts)
        assert len(result) == 1
        assert result[0].content == "New"

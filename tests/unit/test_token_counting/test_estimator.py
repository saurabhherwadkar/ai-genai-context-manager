"""Unit tests for the character-ratio token estimator."""

import pytest

from context_manager.models import Message, Role
from context_manager.token_counting.estimator import EstimatorTokenCounter


class TestEstimatorTokenCounter:
    """Tests for the EstimatorTokenCounter implementation."""

    @pytest.fixture
    def counter(self) -> EstimatorTokenCounter:
        """Create an EstimatorTokenCounter with default ratio."""
        # Use the default 4.0 characters per token ratio
        return EstimatorTokenCounter(ratio=4.0)

    @pytest.mark.asyncio
    async def test_count_single_empty_string(self, counter: EstimatorTokenCounter) -> None:
        """Verify empty string returns zero tokens."""
        # Count tokens for empty input
        result = await counter.count_single("")
        # Should return 0 for empty strings
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_single_short_text(self, counter: EstimatorTokenCounter) -> None:
        """Verify short text returns at least 1 token."""
        # Count tokens for a very short string
        result = await counter.count_single("Hi")
        # Should return at least 1 (2 chars / 4.0 = 0.5, ceil = 1)
        assert result == 1

    @pytest.mark.asyncio
    async def test_count_single_known_length(self, counter: EstimatorTokenCounter) -> None:
        """Verify estimation matches expected calculation."""
        # 20 characters / 4.0 ratio = 5 tokens
        text = "a" * 20
        result = await counter.count_single(text)
        # Should be exactly 5 tokens
        assert result == 5

    @pytest.mark.asyncio
    async def test_count_single_rounds_up(self, counter: EstimatorTokenCounter) -> None:
        """Verify estimation rounds up to avoid underestimation."""
        # 5 characters / 4.0 ratio = 1.25, ceil = 2
        text = "abcde"
        result = await counter.count_single(text)
        # Should round up to 2
        assert result == 2

    @pytest.mark.asyncio
    async def test_count_tokens_multiple_messages(self, counter: EstimatorTokenCounter) -> None:
        """Verify counting tokens for multiple messages."""
        # Create messages with known content lengths
        messages = [
            Message(role=Role.USER, content="a" * 40),
            Message(role=Role.ASSISTANT, content="b" * 80),
        ]
        # Count tokens
        result = await counter.count_tokens(messages)
        # First message: 40/4 + 1 overhead = 11
        assert result.per_message[0] == 11
        # Second message: 80/4 + 1 overhead = 21
        assert result.per_message[1] == 21
        # Total should be sum of per-message counts
        assert result.total == sum(result.per_message)

    @pytest.mark.asyncio
    async def test_count_tokens_empty_list(self, counter: EstimatorTokenCounter) -> None:
        """Verify counting tokens for empty message list."""
        # Count with no messages
        result = await counter.count_tokens([])
        # Should be zero
        assert result.total == 0
        assert result.per_message == []

    def test_invalid_ratio_raises_error(self) -> None:
        """Verify ValueError is raised for non-positive ratio."""
        # Zero ratio should fail
        with pytest.raises(ValueError, match="positive"):
            EstimatorTokenCounter(ratio=0.0)
        # Negative ratio should fail
        with pytest.raises(ValueError, match="positive"):
            EstimatorTokenCounter(ratio=-1.0)

    @pytest.mark.asyncio
    async def test_custom_ratio(self) -> None:
        """Verify a custom ratio changes the estimation."""
        # Create with a ratio of 2.0 (more tokens per character)
        counter = EstimatorTokenCounter(ratio=2.0)
        # 10 characters / 2.0 ratio = 5 tokens
        result = await counter.count_single("a" * 10)
        assert result == 5

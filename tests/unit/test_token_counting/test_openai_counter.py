"""Unit tests for the tiktoken-based OpenAI token counter."""

import pytest

from context_manager.exceptions import TokenCountError
from context_manager.models import Message, Role
from context_manager.token_counting.openai_counter import TiktokenCounter


class TestTiktokenCounter:
    """Tests for the TiktokenCounter implementation."""

    @pytest.fixture
    def counter(self) -> TiktokenCounter:
        """Create a TiktokenCounter with default encoding."""
        # Use the default cl100k_base encoding for testing
        return TiktokenCounter(encoding="cl100k_base")

    @pytest.mark.asyncio
    async def test_count_single_empty_string(self, counter: TiktokenCounter) -> None:
        """Verify empty string returns zero tokens."""
        # Count tokens for an empty string
        result = await counter.count_single("")
        # Empty string should have 0 tokens
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_single_hello(self, counter: TiktokenCounter) -> None:
        """Verify a simple word returns a reasonable token count."""
        # Count tokens for a simple word
        result = await counter.count_single("hello")
        # "hello" should be exactly 1 token in cl100k_base
        assert result == 1

    @pytest.mark.asyncio
    async def test_count_single_sentence(self, counter: TiktokenCounter) -> None:
        """Verify a sentence returns a positive token count."""
        # Count tokens for a typical sentence
        result = await counter.count_single("The quick brown fox jumps over the lazy dog.")
        # Should be a positive number greater than 1
        assert result > 1
        # Should be less than the character count
        assert result < len("The quick brown fox jumps over the lazy dog.")

    @pytest.mark.asyncio
    async def test_count_tokens_single_message(self, counter: TiktokenCounter) -> None:
        """Verify counting tokens for a single message includes overhead."""
        # Create a simple user message
        messages = [Message(role=Role.USER, content="Hello")]
        # Count tokens for the message
        result = await counter.count_tokens(messages)
        # Total should be positive
        assert result.total > 0
        # Should have one per-message count
        assert len(result.per_message) == 1
        # Per-message count should include overhead (more than just "Hello")
        assert result.per_message[0] > 1

    @pytest.mark.asyncio
    async def test_count_tokens_multiple_messages(self, counter: TiktokenCounter) -> None:
        """Verify counting tokens for multiple messages."""
        # Create a multi-message conversation
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="What is Python?"),
            Message(role=Role.ASSISTANT, content="Python is a programming language."),
        ]
        # Count tokens for all messages
        result = await counter.count_tokens(messages)
        # Total should be the sum of per-message plus reply overhead
        assert result.total > sum(result.per_message)
        # Should have a count for each message
        assert len(result.per_message) == 3

    @pytest.mark.asyncio
    async def test_count_tokens_empty_list(self, counter: TiktokenCounter) -> None:
        """Verify counting tokens for an empty message list."""
        # Count tokens for no messages
        result = await counter.count_tokens([])
        # Total should only include the reply overhead (3 tokens)
        assert result.total == 3
        # No per-message counts
        assert result.per_message == []

    def test_invalid_encoding_raises_error(self) -> None:
        """Verify TokenCountError is raised for invalid encoding names."""
        # Attempt to create with a nonexistent encoding
        with pytest.raises(TokenCountError, match="Failed to load"):
            TiktokenCounter(encoding="nonexistent_encoding_xyz")

    @pytest.mark.asyncio
    async def test_deterministic_results(self, counter: TiktokenCounter) -> None:
        """Verify the same input always produces the same token count."""
        # Count the same text twice
        text = "This is a test of deterministic token counting."
        result1 = await counter.count_single(text)
        result2 = await counter.count_single(text)
        # Results should be identical
        assert result1 == result2

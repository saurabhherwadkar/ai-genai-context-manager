"""Unit tests for the Anthropic API-based token counter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from context_manager.models import Message, Role
from context_manager.token_counting.anthropic_counter import AnthropicTokenCounter


class TestAnthropicTokenCounter:
    """Tests for the AnthropicTokenCounter implementation."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock AsyncAnthropic client."""
        # Create a mock client with the expected interface
        client = AsyncMock()
        # Set up the count_tokens response
        response = MagicMock()
        response.input_tokens = 50
        client.messages.count_tokens = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def counter(self, mock_client: AsyncMock) -> AnthropicTokenCounter:
        """Create an AnthropicTokenCounter with mock client."""
        # Initialize with the mock client
        return AnthropicTokenCounter(
            model="claude-sonnet-4-20250514",
            client=mock_client,
            fallback_ratio=4.0,
        )

    @pytest.mark.asyncio
    async def test_count_single_success(
        self, counter: AnthropicTokenCounter, mock_client: AsyncMock
    ) -> None:
        """Verify count_single returns API result on success."""
        # Set up the mock response
        response = MagicMock()
        response.input_tokens = 10
        mock_client.messages.count_tokens = AsyncMock(return_value=response)
        # Count tokens for a single text
        result = await counter.count_single("Hello world")
        # Should return the API's token count
        assert result == 10

    @pytest.mark.asyncio
    async def test_count_single_fallback_on_error(
        self, counter: AnthropicTokenCounter, mock_client: AsyncMock
    ) -> None:
        """Verify count_single falls back to estimator on API failure."""
        # Make the API call raise an exception
        mock_client.messages.count_tokens = AsyncMock(side_effect=Exception("API down"))
        # Count tokens (should fall back to estimator)
        result = await counter.count_single("a" * 40)
        # Estimator with ratio 4.0: 40/4 = 10
        assert result == 10

    @pytest.mark.asyncio
    async def test_count_tokens_success(
        self, counter: AnthropicTokenCounter, mock_client: AsyncMock
    ) -> None:
        """Verify count_tokens returns proportional breakdown on success."""
        # Set up the mock response for the full conversation
        response = MagicMock()
        response.input_tokens = 100
        mock_client.messages.count_tokens = AsyncMock(return_value=response)
        # Create messages with known content lengths
        messages = [
            Message(role=Role.USER, content="a" * 50),
            Message(role=Role.ASSISTANT, content="b" * 50),
        ]
        # Count tokens
        result = await counter.count_tokens(messages)
        # Total should equal the API response
        assert result.total == 100
        # Should have per-message breakdown
        assert len(result.per_message) == 2
        # Sum of per-message should equal total
        assert sum(result.per_message) == 100

    @pytest.mark.asyncio
    async def test_count_tokens_fallback_on_error(
        self, counter: AnthropicTokenCounter, mock_client: AsyncMock
    ) -> None:
        """Verify count_tokens falls back to estimator on API failure."""
        # Make the API call raise an exception
        mock_client.messages.count_tokens = AsyncMock(side_effect=Exception("API error"))
        # Create a simple message
        messages = [Message(role=Role.USER, content="a" * 20)]
        # Count tokens (should fall back)
        result = await counter.count_tokens(messages)
        # Should return estimator results (positive values)
        assert result.total > 0
        assert len(result.per_message) == 1

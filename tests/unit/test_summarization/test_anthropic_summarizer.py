"""Unit tests for the Anthropic summarization strategy."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from context_manager.exceptions import ProviderAuthenticationError, SummarizationError
from context_manager.models import Message, Priority, Role
from context_manager.summarization.anthropic_summarizer import AnthropicSummarizer


class TestAnthropicSummarizer:
    """Tests for the AnthropicSummarizer implementation."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock AsyncAnthropic client."""
        # Create a mock with the expected messages interface
        client = AsyncMock()
        # Set up the response structure
        content_block = MagicMock()
        content_block.text = "This is a summary of the conversation."
        response = MagicMock()
        response.content = [content_block]
        client.messages.create = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def summarizer(self, mock_client: AsyncMock) -> AnthropicSummarizer:
        """Create an AnthropicSummarizer with mock client."""
        # Initialize with the mock client
        return AnthropicSummarizer(
            model="claude-sonnet-4-20250514",
            max_summary_tokens=500,
            client=mock_client,
        )

    @pytest.mark.asyncio
    async def test_summarize_success(
        self, summarizer: AnthropicSummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify successful summarization returns a summary message."""
        # Create messages to summarize
        messages = [
            Message(role=Role.USER, content="What is Python?"),
            Message(role=Role.ASSISTANT, content="Python is a programming language."),
        ]
        # Summarize the messages
        result = await summarizer.summarize(messages)
        # Verify the result structure
        assert isinstance(result, Message)
        assert result.role == Role.SYSTEM
        assert result.priority == Priority.HIGH
        assert "summary" in result.content.lower()
        assert result.metadata["source"] == "summarization"

    @pytest.mark.asyncio
    async def test_summarize_empty_list_raises_error(
        self, summarizer: AnthropicSummarizer
    ) -> None:
        """Verify summarizing empty messages raises SummarizationError."""
        # Attempt to summarize empty list
        with pytest.raises(SummarizationError, match="empty"):
            await summarizer.summarize([])

    @pytest.mark.asyncio
    async def test_summarize_api_failure_raises_error(
        self, summarizer: AnthropicSummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify API failures are wrapped in SummarizationError."""
        # Make the API call fail
        mock_client.messages.create = AsyncMock(side_effect=Exception("Network error"))
        # Create messages to summarize
        messages = [Message(role=Role.USER, content="Test")]
        # Should raise SummarizationError
        with pytest.raises(SummarizationError, match="failed"):
            await summarizer.summarize(messages)

    @pytest.mark.asyncio
    async def test_summarize_auth_error(
        self, summarizer: AnthropicSummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify authentication errors raise ProviderAuthenticationError."""
        # Make the API call fail with auth error
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("authentication error: invalid api_key")
        )
        # Create messages to summarize
        messages = [Message(role=Role.USER, content="Test")]
        # Should raise ProviderAuthenticationError
        with pytest.raises(ProviderAuthenticationError):
            await summarizer.summarize(messages)

    @pytest.mark.asyncio
    async def test_summarize_empty_content_raises_error(
        self, summarizer: AnthropicSummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify empty API response raises SummarizationError."""
        # Set up response with no content blocks
        response = MagicMock()
        response.content = []
        mock_client.messages.create = AsyncMock(return_value=response)
        # Create messages
        messages = [Message(role=Role.USER, content="Test")]
        # Should raise SummarizationError
        with pytest.raises(SummarizationError, match="empty"):
            await summarizer.summarize(messages)

    @pytest.mark.asyncio
    async def test_summarize_calls_api_with_correct_model(
        self, summarizer: AnthropicSummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify the API is called with the configured model."""
        # Create messages to summarize
        messages = [Message(role=Role.USER, content="Hello")]
        # Summarize
        await summarizer.summarize(messages)
        # Verify the API was called with correct parameters
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 500

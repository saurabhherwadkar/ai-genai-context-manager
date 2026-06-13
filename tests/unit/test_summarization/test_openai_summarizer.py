"""Unit tests for the OpenAI summarization strategy."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from context_manager.exceptions import ProviderAuthenticationError, SummarizationError
from context_manager.models import Message, Priority, Role
from context_manager.summarization.openai_summarizer import OpenAISummarizer


class TestOpenAISummarizer:
    """Tests for the OpenAISummarizer implementation."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock AsyncOpenAI client."""
        # Create a mock with the expected chat completions interface
        client = AsyncMock()
        # Set up the response structure
        choice = MagicMock()
        choice.message.content = "This is a summary of the conversation."
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def summarizer(self, mock_client: AsyncMock) -> OpenAISummarizer:
        """Create an OpenAISummarizer with mock client."""
        # Initialize with the mock client
        return OpenAISummarizer(
            model="gpt-4o-mini",
            max_summary_tokens=500,
            client=mock_client,
        )

    @pytest.mark.asyncio
    async def test_summarize_success(
        self, summarizer: OpenAISummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify successful summarization returns a summary message."""
        # Create messages to summarize
        messages = [
            Message(role=Role.USER, content="What is Python?"),
            Message(role=Role.ASSISTANT, content="Python is a programming language."),
        ]
        # Summarize the messages
        result = await summarizer.summarize(messages)
        # Verify the result is a Message
        assert isinstance(result, Message)
        # Verify it has the summary content
        assert "summary" in result.content.lower()
        # Verify it's a system message with high priority
        assert result.role == Role.SYSTEM
        assert result.priority == Priority.HIGH
        # Verify metadata tracks the source
        assert result.metadata["source"] == "summarization"
        assert result.metadata["original_count"] == "2"

    @pytest.mark.asyncio
    async def test_summarize_empty_list_raises_error(
        self, summarizer: OpenAISummarizer
    ) -> None:
        """Verify summarizing empty messages raises SummarizationError."""
        # Attempt to summarize empty list
        with pytest.raises(SummarizationError, match="empty"):
            await summarizer.summarize([])

    @pytest.mark.asyncio
    async def test_summarize_api_failure_raises_error(
        self, summarizer: OpenAISummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify API failures are wrapped in SummarizationError."""
        # Make the API call fail
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        # Create messages to summarize
        messages = [Message(role=Role.USER, content="Test")]
        # Should raise SummarizationError
        with pytest.raises(SummarizationError, match="failed"):
            await summarizer.summarize(messages)

    @pytest.mark.asyncio
    async def test_summarize_auth_error(
        self, summarizer: OpenAISummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify authentication errors raise ProviderAuthenticationError."""
        # Make the API call fail with auth error
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Authentication failed: invalid API key")
        )
        # Create messages to summarize
        messages = [Message(role=Role.USER, content="Test")]
        # Should raise ProviderAuthenticationError
        with pytest.raises(ProviderAuthenticationError):
            await summarizer.summarize(messages)

    @pytest.mark.asyncio
    async def test_summarize_empty_response_raises_error(
        self, summarizer: OpenAISummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify empty API response raises SummarizationError."""
        # Set up response with empty content
        choice = MagicMock()
        choice.message.content = ""
        response = MagicMock()
        response.choices = [choice]
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        # Create messages
        messages = [Message(role=Role.USER, content="Test")]
        # Should raise SummarizationError
        with pytest.raises(SummarizationError, match="empty"):
            await summarizer.summarize(messages)

    @pytest.mark.asyncio
    async def test_summarize_calls_api_with_correct_params(
        self, summarizer: OpenAISummarizer, mock_client: AsyncMock
    ) -> None:
        """Verify the API is called with expected parameters."""
        # Create messages to summarize
        messages = [Message(role=Role.USER, content="Hello world")]
        # Summarize
        await summarizer.summarize(messages)
        # Verify the API was called
        mock_client.chat.completions.create.assert_called_once()
        # Get the call arguments
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        # Verify key parameters
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.3

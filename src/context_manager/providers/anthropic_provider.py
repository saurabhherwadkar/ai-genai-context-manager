"""Anthropic provider wrapper for the LLMProvider protocol.

Wraps the Anthropic messages API to provide a uniform
interface for sending messages and receiving completions.
"""

import logging
from collections.abc import AsyncIterator, Sequence

from anthropic import AsyncAnthropic

from context_manager.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from context_manager.models import Message, Role

# Module-level logger for provider diagnostics
logger = logging.getLogger(__name__)

# Default maximum tokens for Anthropic response generation
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider:
    """LLM provider implementation for Anthropic's messages API.

    Supports both synchronous completions and streaming responses
    with proper error handling and system message extraction.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        client: AsyncAnthropic | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            model: Anthropic model ID for completions.
            client: Optional pre-configured AsyncAnthropic client.
            timeout: Request timeout in seconds.
            max_retries: Maximum retries on transient failures.
            max_tokens: Maximum tokens for response generation.
        """
        # Store the model identifier for API calls
        self._model = model
        # Use provided client or create with retry configuration
        self._client = client or AsyncAnthropic(max_retries=max_retries)
        # Store the timeout for request limits
        self._timeout = timeout
        # Store the maximum response tokens
        self._max_tokens = max_tokens
        # Log provider initialization
        logger.debug("AnthropicProvider initialized (model=%s, timeout=%.1fs)", model, timeout)

    async def complete(self, messages: Sequence[Message], **kwargs: object) -> str:
        """Send messages and return the complete response text.

        Anthropic requires system messages to be passed separately
        from the conversation messages.

        Args:
            messages: Sequence of conversation messages.
            **kwargs: Additional parameters passed to the API.

        Returns:
            The complete response text from the model.

        Raises:
            ProviderError: If the API call fails.
        """
        # Separate system messages from conversation messages
        system_text, conversation = self._separate_system_messages(messages)
        # Log the completion request
        logger.debug("Requesting completion with %d messages", len(conversation))
        # Call the Anthropic messages API
        try:
            # Build the API call parameters
            api_kwargs: dict[str, object] = {
                "model": self._model,
                "messages": conversation,
                "max_tokens": self._max_tokens,
            }
            # Add system parameter only if system messages exist
            if system_text:
                api_kwargs["system"] = system_text
            # Make the non-streaming API call
            response = await self._client.messages.create(**api_kwargs)  # type: ignore[arg-type]
            # Extract text from the first content block
            if not response.content:
                return ""
            # Get the text content from the response
            content = response.content[0].text
            # Log successful completion
            logger.debug("Completion received: %d chars", len(content))
            # Return the response content string
            return content
        except Exception as exc:
            # Categorize and re-raise the error appropriately
            self._handle_error(exc)

    async def complete_stream(
        self, messages: Sequence[Message], **kwargs: object
    ) -> AsyncIterator[str]:
        """Send messages and yield response chunks as they arrive.

        Args:
            messages: Sequence of conversation messages.
            **kwargs: Additional parameters passed to the API.

        Yields:
            Text chunks as they are received from the API.

        Raises:
            ProviderError: If the API call fails.
        """
        # Separate system messages from conversation messages
        system_text, conversation = self._separate_system_messages(messages)
        # Log the streaming request
        logger.debug("Requesting streaming completion with %d messages", len(conversation))
        # Call the Anthropic API with streaming
        try:
            # Build the API call parameters
            api_kwargs: dict[str, object] = {
                "model": self._model,
                "messages": conversation,
                "max_tokens": self._max_tokens,
            }
            # Add system parameter only if system messages exist
            if system_text:
                api_kwargs["system"] = system_text
            # Create the streaming response using the stream manager
            async with self._client.messages.stream(**api_kwargs) as stream:  # type: ignore[arg-type]
                # Yield each text chunk as it arrives
                async for text in stream.text_stream:
                    # Only yield non-empty text chunks
                    if text:
                        yield text
        except Exception as exc:
            # Categorize and re-raise the error appropriately
            self._handle_error(exc)

    def _separate_system_messages(
        self, messages: Sequence[Message]
    ) -> tuple[str, list[dict[str, str]]]:
        """Separate system messages from conversation messages.

        Anthropic's API requires system messages to be passed as a
        separate parameter rather than inline with conversation messages.

        Args:
            messages: Full sequence of messages including system.

        Returns:
            Tuple of (system_text, conversation_messages).
        """
        # Collect system message content separately
        system_parts: list[str] = []
        # Collect non-system messages for the conversation parameter
        conversation: list[dict[str, str]] = []
        # Partition messages by role
        for message in messages:
            # System messages go to the system parameter
            if message.role == Role.SYSTEM:
                system_parts.append(message.content)
            else:
                # Non-system messages go to the messages parameter
                conversation.append({"role": message.role.value, "content": message.content})
        # Join all system messages with newlines
        system_text = "\n".join(system_parts)
        # Return the separated system text and conversation
        return system_text, conversation

    def _handle_error(self, exc: Exception) -> None:
        """Categorize an exception and raise the appropriate provider error.

        Args:
            exc: The original exception from the API call.

        Raises:
            ProviderAuthenticationError: For auth failures.
            ProviderRateLimitError: For rate limit errors.
            ProviderTimeoutError: For timeout errors.
            ProviderError: For all other failures.
        """
        # Convert the exception message to lowercase for matching
        error_message = str(exc).lower()
        # Check for authentication errors
        if "authentication" in error_message or "api_key" in error_message:
            raise ProviderAuthenticationError(f"Anthropic auth failed: {exc}") from exc
        # Check for rate limit errors
        if "rate limit" in error_message or "429" in error_message:
            raise ProviderRateLimitError(f"Anthropic rate limited: {exc}") from exc
        # Check for timeout errors
        if "timeout" in error_message or "timed out" in error_message:
            raise ProviderTimeoutError(f"Anthropic request timed out: {exc}") from exc
        # Default to generic provider error for unrecognized failures
        raise ProviderError(f"Anthropic API error: {exc}") from exc

"""OpenAI provider wrapper for the LLMProvider protocol.

Wraps the OpenAI chat completions API to provide a uniform
interface for sending messages and receiving completions.
"""

import logging
from collections.abc import AsyncIterator, Sequence

from openai import AsyncOpenAI

from context_manager.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from context_manager.models import Message

# Module-level logger for provider diagnostics
logger = logging.getLogger(__name__)


class OpenAIProvider:
    """LLM provider implementation for OpenAI's chat completions API.

    Supports both synchronous completions and streaming responses
    with proper error handling and retry categorization.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        client: AsyncOpenAI | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            model: OpenAI model ID for completions.
            client: Optional pre-configured AsyncOpenAI client.
            timeout: Request timeout in seconds.
            max_retries: Maximum retries on transient failures.
        """
        # Store the model identifier for API calls
        self._model = model
        # Use provided client or create with retry configuration
        self._client = client or AsyncOpenAI(max_retries=max_retries)
        # Store the timeout for request limits
        self._timeout = timeout
        # Log provider initialization
        logger.debug("OpenAIProvider initialized (model=%s, timeout=%.1fs)", model, timeout)

    async def complete(self, messages: Sequence[Message], **kwargs: object) -> str:
        """Send messages and return the complete response text.

        Args:
            messages: Sequence of conversation messages.
            **kwargs: Additional parameters passed to the API.

        Returns:
            The complete response text from the model.

        Raises:
            ProviderError: If the API call fails.
        """
        # Format messages into the OpenAI API format
        formatted = self._format_messages(messages)
        # Log the completion request with message count
        logger.debug("Requesting completion with %d messages", len(messages))
        # Call the OpenAI chat completions API
        try:
            # Make the non-streaming API call
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=formatted,
                timeout=self._timeout,
                **kwargs,  # type: ignore[arg-type]
            )
            # Extract the response text from the first choice
            content = response.choices[0].message.content or ""
            # Log successful completion with response length
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
        # Format messages into the OpenAI API format
        formatted = self._format_messages(messages)
        # Log the streaming request
        logger.debug("Requesting streaming completion with %d messages", len(messages))
        # Call the OpenAI API with streaming enabled
        try:
            # Create the streaming response
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=formatted,
                stream=True,
                timeout=self._timeout,
                **kwargs,  # type: ignore[arg-type]
            )
            # Yield each text chunk as it arrives from the stream
            async for chunk in stream:
                # Extract delta content from the streaming chunk
                delta = chunk.choices[0].delta.content
                # Only yield non-empty content deltas
                if delta:
                    yield delta
        except Exception as exc:
            # Categorize and re-raise the error appropriately
            self._handle_error(exc)

    def _format_messages(self, messages: Sequence[Message]) -> list[dict[str, str]]:
        """Convert Message objects to OpenAI API message format.

        Args:
            messages: Sequence of Message objects to format.

        Returns:
            List of dictionaries with role and content keys.
        """
        # Map each Message to the dict format expected by the API
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

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
        if "authentication" in error_message or "api key" in error_message:
            raise ProviderAuthenticationError(f"OpenAI auth failed: {exc}") from exc
        # Check for rate limit errors
        if "rate limit" in error_message or "429" in error_message:
            raise ProviderRateLimitError(f"OpenAI rate limited: {exc}") from exc
        # Check for timeout errors
        if "timeout" in error_message or "timed out" in error_message:
            raise ProviderTimeoutError(f"OpenAI request timed out: {exc}") from exc
        # Default to generic provider error for unrecognized failures
        raise ProviderError(f"OpenAI API error: {exc}") from exc

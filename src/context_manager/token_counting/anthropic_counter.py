"""Anthropic API-based token counter.

Uses Anthropic's count_tokens API endpoint for accurate token counting.
Falls back to a character-ratio estimator when the API is unavailable.
"""

import logging
from collections.abc import Sequence

from anthropic import AsyncAnthropic

from context_manager.exceptions import TokenCountError
from context_manager.models import Message, TokenCount
from context_manager.token_counting.estimator import EstimatorTokenCounter

# Module-level logger for token counting diagnostics
logger = logging.getLogger(__name__)


class AnthropicTokenCounter:
    """Token counter using Anthropic's count_tokens API endpoint.

    Provides accurate token counting for Anthropic models by calling
    the official API. Falls back to estimation on API failures.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        client: AsyncAnthropic | None = None,
        fallback_ratio: float = 4.0,
    ) -> None:
        """Initialize the Anthropic token counter.

        Args:
            model: Anthropic model ID for the count_tokens endpoint.
            client: Optional pre-configured AsyncAnthropic client.
            fallback_ratio: Characters-per-token ratio for the fallback estimator.
        """
        # Store the model ID for API calls
        self._model = model
        # Use provided client or create a new one from environment API key
        self._client = client or AsyncAnthropic()
        # Create a fallback estimator for when the API is unavailable
        self._fallback = EstimatorTokenCounter(ratio=fallback_ratio)
        # Log initialization with the configured model
        logger.debug("AnthropicTokenCounter initialized for model: %s", model)

    async def count_tokens(self, messages: Sequence[Message]) -> TokenCount:
        """Count tokens using the Anthropic count_tokens API endpoint.

        Falls back to the character-ratio estimator if the API call fails.

        Args:
            messages: Sequence of Message objects to count.

        Returns:
            TokenCount with total and per-message breakdown.
        """
        # Attempt to count tokens via the Anthropic API
        try:
            # Build per-message counts by calling the API for each message
            per_message_counts = await self._count_via_api(messages)
            # Calculate total from the individual message counts
            total = sum(per_message_counts)
            # Log successful API-based counting
            logger.debug("API counted %d total tokens for %d messages", total, len(messages))
            # Return the structured token count result
            return TokenCount(total=total, per_message=per_message_counts)
        except Exception as exc:
            # Log the fallback with a warning for visibility
            logger.warning("Anthropic token count API failed, using estimator: %s", exc)
            # Delegate to the fallback estimator for approximate counts
            return await self._fallback.count_tokens(messages)

    async def count_single(self, text: str) -> int:
        """Count tokens for a single text string via the API.

        Falls back to estimation if the API is unavailable.

        Args:
            text: The text string to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        # Attempt API-based counting for the single text
        try:
            # Call the Anthropic count_tokens endpoint with a user message
            response = await self._client.messages.count_tokens(
                model=self._model,
                messages=[{"role": "user", "content": text}],
            )
            # Extract the token count from the API response
            return response.input_tokens
        except Exception as exc:
            # Log the fallback usage at warning level
            logger.warning("Single token count API failed, using estimator: %s", exc)
            # Delegate to the fallback estimator
            return await self._fallback.count_single(text)

    async def _count_via_api(self, messages: Sequence[Message]) -> list[int]:
        """Count tokens for each message individually via the API.

        Args:
            messages: Sequence of messages to count.

        Returns:
            List of token counts, one per message.

        Raises:
            TokenCountError: If the API call fails for any message.
        """
        # Initialize the per-message counts list
        per_message_counts: list[int] = []
        # Count the full conversation to get the total
        formatted_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
        # Call the API with the full message list
        try:
            # Send all messages at once for an accurate cumulative count
            full_response = await self._client.messages.count_tokens(
                model=self._model,
                messages=formatted_messages,
            )
            # Store the full conversation token count
            full_count = full_response.input_tokens
        except Exception as exc:
            # Wrap API errors in our custom exception type
            raise TokenCountError(f"Anthropic count_tokens API failed: {exc}") from exc
        # Estimate per-message breakdown proportionally from the total
        for message in messages:
            # Use character length ratio for proportional breakdown
            char_ratio = len(message.content) / max(
                sum(len(m.content) for m in messages), 1
            )
            # Compute this message's estimated share of the total
            estimated_count = max(1, int(full_count * char_ratio))
            # Append the proportional estimate to the list
            per_message_counts.append(estimated_count)
        # Adjust the last message to ensure counts sum to the exact total
        if per_message_counts:
            # Calculate the difference between sum and actual total
            difference = full_count - sum(per_message_counts)
            # Apply the adjustment to the last message count
            per_message_counts[-1] = max(1, per_message_counts[-1] + difference)
        # Return the per-message breakdown list
        return per_message_counts

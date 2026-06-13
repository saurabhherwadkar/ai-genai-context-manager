"""Tiktoken-based token counter for OpenAI models.

Uses OpenAI's tiktoken library for fast, accurate local token counting
without requiring any API calls. Supports all OpenAI model encodings.
"""

import logging
from collections.abc import Sequence

import tiktoken

from context_manager.exceptions import TokenCountError
from context_manager.models import Message, TokenCount

# Module-level logger for token counting diagnostics
logger = logging.getLogger(__name__)

# Overhead tokens added by the ChatML format per message
TOKENS_PER_MESSAGE_OVERHEAD = 4

# Tokens added at the end of the conversation for reply priming
TOKENS_REPLY_OVERHEAD = 3


class TiktokenCounter:
    """Token counter using OpenAI's tiktoken encoding library.

    Provides fast, deterministic local token counting for any
    model that uses a tiktoken-compatible encoding scheme.
    """

    def __init__(self, encoding: str = "cl100k_base") -> None:
        """Initialize the counter with a specific tiktoken encoding.

        Args:
            encoding: Name of the tiktoken encoding to use.

        Raises:
            TokenCountError: If the specified encoding cannot be loaded.
        """
        # Store the encoding name for logging and diagnostics
        self._encoding_name = encoding
        # Attempt to load the tiktoken encoding by name
        try:
            # Get the encoding instance for token conversion
            self._encoding = tiktoken.get_encoding(encoding)
        except Exception as exc:
            # Wrap encoding load failures in our custom exception
            raise TokenCountError(
                f"Failed to load tiktoken encoding '{encoding}': {exc}"
            ) from exc
        # Log successful initialization at debug level
        logger.debug("TiktokenCounter initialized with encoding: %s", encoding)

    async def count_tokens(self, messages: Sequence[Message]) -> TokenCount:
        """Count tokens for a sequence of messages including ChatML overhead.

        Accounts for the per-message overhead tokens that the API adds
        when formatting messages in the ChatML conversation format.

        Args:
            messages: Sequence of Message objects to count.

        Returns:
            TokenCount with total and per-message breakdown.
        """
        # Initialize the list to store per-message token counts
        per_message_counts: list[int] = []
        # Iterate through each message to compute its token count
        for message in messages:
            # Count content tokens plus the ChatML per-message overhead
            message_tokens = self._count_message_tokens(message)
            # Append this message's count to the per-message list
            per_message_counts.append(message_tokens)
        # Calculate the total including the reply priming overhead
        total = sum(per_message_counts) + TOKENS_REPLY_OVERHEAD
        # Log the computed total at debug level for diagnostics
        logger.debug("Counted %d total tokens for %d messages", total, len(messages))
        # Return the structured token count result
        return TokenCount(total=total, per_message=per_message_counts)

    async def count_single(self, text: str) -> int:
        """Count tokens for a single text string without message overhead.

        Args:
            text: The text string to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        # Encode the text to get the token list and return its length
        token_ids = self._encoding.encode(text)
        # Return the count of token IDs produced by encoding
        return len(token_ids)

    def _count_message_tokens(self, message: Message) -> int:
        """Count tokens for a single message including role and overhead.

        The ChatML format adds overhead tokens for each message:
        <|start|>role\ncontent<|end|> which costs extra tokens.

        Args:
            message: The Message to count tokens for.

        Returns:
            Total tokens for this message including formatting overhead.
        """
        # Start with the per-message overhead tokens for ChatML format
        token_count = TOKENS_PER_MESSAGE_OVERHEAD
        # Add tokens for the role field (e.g., "user", "assistant")
        token_count += len(self._encoding.encode(message.role.value))
        # Add tokens for the actual message content
        token_count += len(self._encoding.encode(message.content))
        # Return the total including overhead and content
        return token_count

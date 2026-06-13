"""Protocol definitions for all pluggable strategy interfaces.

Protocols use structural typing so third-party implementations
need not inherit from these classes — any object with matching
method signatures satisfies the protocol at runtime.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from context_manager.models import Message, TokenCount


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens in messages.

    Implementations may use local tokenizers (tiktoken) or
    remote API endpoints (Anthropic count_tokens).
    """

    async def count_tokens(self, messages: Sequence[Message]) -> TokenCount:
        """Count tokens for a sequence of messages.

        Returns a TokenCount with both the total and per-message breakdown.
        """
        ...  # pragma: no cover

    async def count_single(self, text: str) -> int:
        """Count tokens for a single text string.

        Useful for checking individual message sizes before adding.
        """
        ...  # pragma: no cover


@runtime_checkable
class TrimmingStrategy(Protocol):
    """Protocol for trimming messages to fit within a token budget.

    Implementations decide which messages to remove based on
    their specific strategy (FIFO, sliding window, priority).
    """

    def trim(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counts: list[int],
    ) -> list[Message]:
        """Trim messages to fit within the specified token budget.

        Args:
            messages: The full list of messages to trim.
            max_tokens: The maximum allowed total token count.
            token_counts: Pre-computed token count for each message.

        Returns:
            A new list containing only the retained messages.
        """
        ...  # pragma: no cover


@runtime_checkable
class SummarizationStrategy(Protocol):
    """Protocol for summarizing messages to reduce token usage.

    Summarization preserves semantic content while reducing
    the number of tokens consumed by older messages.
    """

    async def summarize(self, messages: Sequence[Message]) -> Message:
        """Summarize a sequence of messages into a single condensed message.

        The returned message should capture the key context and decisions
        from the input messages in a more compact form.
        """
        ...  # pragma: no cover


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider API interactions.

    Abstracts over different LLM APIs (OpenAI, Anthropic) to
    provide a uniform interface for completions and streaming.
    """

    async def complete(self, messages: Sequence[Message], **kwargs: object) -> str:
        """Send messages to the LLM and return the complete response text.

        This is a non-streaming call that waits for the full response.
        """
        ...  # pragma: no cover

    async def complete_stream(
        self, messages: Sequence[Message], **kwargs: object
    ) -> AsyncIterator[str]:
        """Send messages to the LLM and yield response chunks as they arrive.

        Enables real-time streaming of the LLM's response for lower latency.
        """
        ...  # pragma: no cover

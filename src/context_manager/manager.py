"""Context manager orchestrator that coordinates trimming, summarization, and token counting.

This is the main entry point for the context management system.
It composes token counting, trimming, and summarization strategies
to automatically manage conversation context windows.
"""

import asyncio
import logging
import re
from collections.abc import Sequence

from context_manager.config import ContextManagerConfig
from context_manager.exceptions import (
    SummarizationError,
    TokenBudgetExceededError,
    TrimmingError,
    ValidationError,
)
from context_manager.models import ConversationState, Message, Priority, Role, TokenCount
from context_manager.protocols import SummarizationStrategy, TokenCounter, TrimmingStrategy

# Module-level logger for context management diagnostics
logger = logging.getLogger(__name__)

# Regex pattern matching control characters to sanitize from input
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ContextManager:
    """Orchestrates token counting, trimming, and summarization strategies.

    The central coordinator that manages conversation context windows by
    composing injected strategy implementations. Handles message validation,
    token tracking, and automatic context window management.
    """

    def __init__(
        self,
        config: ContextManagerConfig,
        token_counter: TokenCounter,
        trimming_strategy: TrimmingStrategy,
        summarization_strategy: SummarizationStrategy | None = None,
    ) -> None:
        """Initialize the context manager with injected dependencies.

        Args:
            config: Application configuration with limits and settings.
            token_counter: Strategy for counting tokens in messages.
            trimming_strategy: Strategy for removing messages to fit budget.
            summarization_strategy: Optional strategy for summarizing messages.
        """
        # Store the configuration for limit checking and settings
        self._config = config
        # Store the token counter strategy for measuring messages
        self._token_counter = token_counter
        # Store the trimming strategy for removing excess messages
        self._trimming_strategy = trimming_strategy
        # Store the optional summarization strategy
        self._summarization_strategy = summarization_strategy
        # Initialize the conversation state with configured limits
        self._state = ConversationState(
            max_tokens=config.provider.max_context_tokens,
            reserved_tokens=config.provider.reserved_response_tokens,
        )
        # Log successful initialization with strategy details
        logger.info(
            "ContextManager initialized (max_tokens=%d, reserved=%d)",
            config.provider.max_context_tokens,
            config.provider.reserved_response_tokens,
        )

    async def add_message(self, message: Message) -> ConversationState:
        """Add a single message to the conversation and manage the context window.

        Validates the message, counts its tokens, adds it to the state,
        and triggers trimming or summarization if the budget is exceeded.

        Args:
            message: The message to add to the conversation.

        Returns:
            The updated conversation state after adding the message.

        Raises:
            ValidationError: If the message fails validation checks.
            TokenBudgetExceededError: If the message alone exceeds the budget.
        """
        # Validate the message content against security constraints
        validated_message = self._validate_message(message)
        # Count tokens for this individual message
        token_count = await self._token_counter.count_single(validated_message.content)
        # Calculate the effective budget excluding reserved tokens
        effective_budget = self._state.max_tokens - self._state.reserved_tokens
        # Check if this single message exceeds the entire budget
        if token_count > effective_budget:
            # A single message that can't fit is an unrecoverable error
            raise TokenBudgetExceededError(
                token_count=token_count,
                budget=effective_budget,
            )
        # Create a new message with the cached token count
        counted_message = Message(
            role=validated_message.role,
            content=validated_message.content,
            priority=validated_message.priority,
            timestamp=validated_message.timestamp,
            token_count=token_count,
            metadata=validated_message.metadata,
        )
        # Append the message to the conversation state
        self._state.messages.append(counted_message)
        # Update the running token total
        self._state.total_tokens += token_count
        # Log the message addition at debug level
        logger.debug(
            "Added %s message (%d tokens, total: %d)",
            message.role.value,
            token_count,
            self._state.total_tokens,
        )
        # Trigger context management if over budget
        if self._state.is_over_budget:
            # Attempt summarization first if enabled, then fall back to trimming
            await self._manage_context()
        # Return the current conversation state
        return self._state

    async def add_messages(self, messages: Sequence[Message]) -> ConversationState:
        """Add multiple messages to the conversation sequentially.

        Processes each message in order, triggering context management
        as needed between additions.

        Args:
            messages: Sequence of messages to add in order.

        Returns:
            The updated conversation state after all messages are added.
        """
        # Process each message sequentially to maintain ordering
        for message in messages:
            # Add each message individually to trigger per-message validation
            await self.add_message(message)
        # Return the final conversation state
        return self._state

    async def get_context_window(self) -> list[Message]:
        """Get the current context window messages, trimmed to fit.

        Returns the messages that should be sent to the LLM, ensuring
        they fit within the configured token budget.

        Returns:
            List of messages fitting within the context window.
        """
        # Ensure the context is within budget before returning
        if self._state.is_over_budget:
            # Manage context to bring it within budget
            await self._manage_context()
        # Return a copy of the current messages list
        return list(self._state.messages)

    async def trim_to_fit(self) -> list[Message]:
        """Force trimming of the current messages to fit the token budget.

        Applies the configured trimming strategy regardless of whether
        the current context exceeds the budget.

        Returns:
            The trimmed list of messages.
        """
        # Get the effective token budget for trimming
        effective_budget = self._state.max_tokens - self._state.reserved_tokens
        # Get token counts for all current messages
        token_counts = await self._get_message_token_counts()
        # Apply the trimming strategy to reduce messages
        trimmed = self._trimming_strategy.trim(
            self._state.messages,
            effective_budget,
            token_counts,
        )
        # Calculate how many messages were removed
        removed_count = len(self._state.messages) - len(trimmed)
        # Update the state with the trimmed messages
        self._state.messages = trimmed
        # Update the trimmed counter for tracking
        self._state.trimmed_count += removed_count
        # Recalculate the total token count after trimming
        await self._recalculate_tokens()
        # Log the trimming result
        logger.info("Trimmed %d messages, %d remaining", removed_count, len(trimmed))
        # Return the trimmed message list
        return trimmed

    async def summarize_and_trim(self) -> list[Message]:
        """Summarize older messages and trim the conversation.

        First attempts to summarize older messages into a condensed form,
        then applies trimming if still over budget. Falls back to pure
        trimming if summarization fails.

        Returns:
            The resulting message list after summarization and trimming.
        """
        # Check if summarization is available and enabled
        if not self._summarization_strategy or not self._config.summarization.enabled:
            # Fall back to pure trimming if summarization is not available
            logger.debug("Summarization not available, falling back to trimming")
            return await self.trim_to_fit()
        # Identify messages eligible for summarization (non-critical, non-system)
        summarizable = self._get_summarizable_messages()
        # Need at least 2 messages to make summarization worthwhile
        if len(summarizable) < 2:
            # Not enough messages to summarize, just trim
            return await self.trim_to_fit()
        # Attempt summarization of the eligible messages
        try:
            # Call the summarization strategy with eligible messages
            summary_message = await self._summarization_strategy.summarize(summarizable)
            # Remove the summarized messages from the conversation
            self._replace_with_summary(summarizable, summary_message)
            # Update the summarized counter
            self._state.summarized_count += len(summarizable)
            # Recalculate tokens after the replacement
            await self._recalculate_tokens()
            # Log successful summarization
            logger.info(
                "Summarized %d messages into 1 summary message",
                len(summarizable),
            )
        except SummarizationError as exc:
            # Log the summarization failure and fall back to trimming
            logger.warning("Summarization failed, falling back to trimming: %s", exc)
            return await self.trim_to_fit()
        # Apply trimming if still over budget after summarization
        if self._state.is_over_budget:
            return await self.trim_to_fit()
        # Return the current messages after successful summarization
        return list(self._state.messages)

    def get_token_usage(self) -> TokenCount:
        """Get the current token usage breakdown.

        Returns:
            TokenCount with total and per-message token counts.
        """
        # Build per-message counts from cached values on each message
        per_message = [msg.token_count or 0 for msg in self._state.messages]
        # Return the structured token count
        return TokenCount(total=self._state.total_tokens, per_message=per_message)

    def get_state(self) -> ConversationState:
        """Get the current conversation state.

        Returns:
            The current ConversationState with all tracking information.
        """
        # Return the mutable state object directly
        return self._state

    def reset(self) -> None:
        """Reset the conversation state, clearing all messages and counters.

        Creates a fresh state while preserving the configured limits.
        """
        # Re-initialize the state with the same configuration
        self._state = ConversationState(
            max_tokens=self._config.provider.max_context_tokens,
            reserved_tokens=self._config.provider.reserved_response_tokens,
        )
        # Log the reset action
        logger.info("Context manager state reset")

    def add_message_sync(self, message: Message) -> ConversationState:
        """Synchronous wrapper for add_message.

        Convenience method for callers that cannot use async/await.

        Args:
            message: The message to add.

        Returns:
            The updated conversation state.
        """
        # Run the async method in a new event loop
        return asyncio.run(self.add_message(message))

    def get_context_window_sync(self) -> list[Message]:
        """Synchronous wrapper for get_context_window.

        Convenience method for callers that cannot use async/await.

        Returns:
            List of messages in the current context window.
        """
        # Run the async method in a new event loop
        return asyncio.run(self.get_context_window())

    async def _manage_context(self) -> None:
        """Manage the context window by summarizing or trimming.

        Called automatically when the conversation exceeds the token budget.
        Tries summarization first if enabled, then falls back to trimming.
        """
        # Log the context management trigger
        logger.debug(
            "Context management triggered (tokens: %d, budget: %d)",
            self._state.total_tokens,
            self._state.max_tokens - self._state.reserved_tokens,
        )
        # Try summarization if available and enabled
        if self._summarization_strategy and self._config.summarization.enabled:
            # Attempt the combined summarize-and-trim approach
            await self.summarize_and_trim()
        else:
            # Fall back to pure trimming
            await self.trim_to_fit()
        # Verify that trimming actually reduced the context enough
        if self._state.is_over_budget:
            # Log the error condition where we can't fit within budget
            logger.error(
                "Cannot fit within budget after trimming (tokens: %d, budget: %d)",
                self._state.total_tokens,
                self._state.max_tokens - self._state.reserved_tokens,
            )
            # Raise an error to signal the unrecoverable condition
            raise TrimmingError(
                f"Cannot reduce context to fit budget: "
                f"{self._state.total_tokens} tokens remain, "
                f"budget is {self._state.max_tokens - self._state.reserved_tokens}"
            )

    async def _get_message_token_counts(self) -> list[int]:
        """Get or compute token counts for all current messages.

        Uses cached counts from messages where available, falling back
        to the token counter for any messages without cached counts.

        Returns:
            List of token counts, one per message.
        """
        # Build the counts list using cached values where possible
        counts: list[int] = []
        # Check each message for a cached token count
        for message in self._state.messages:
            # Use the cached count if available
            if message.token_count is not None:
                counts.append(message.token_count)
            else:
                # Compute the count for messages without cached values
                count = await self._token_counter.count_single(message.content)
                counts.append(count)
        # Return the complete list of token counts
        return counts

    async def _recalculate_tokens(self) -> None:
        """Recalculate the total token count from all current messages.

        Called after trimming or summarization to update the state's
        total_tokens to reflect the current message list.
        """
        # Get fresh token counts for all messages
        counts = await self._get_message_token_counts()
        # Update the state total with the sum of all counts
        self._state.total_tokens = sum(counts)

    def _get_summarizable_messages(self) -> list[Message]:
        """Identify messages eligible for summarization.

        Returns older messages that are not system messages and not
        critical priority — these are safe to condense.

        Returns:
            List of messages that can be summarized.
        """
        # Filter for messages that can be safely summarized
        summarizable: list[Message] = []
        # Keep the most recent messages unsummarized for continuity
        keep_recent = min(10, len(self._state.messages) // 2)
        # Only consider older messages for summarization
        candidates = self._state.messages[:-keep_recent] if keep_recent > 0 else []
        # Filter candidates based on priority and role
        for message in candidates:
            # Skip system messages as they contain critical instructions
            if message.role == Role.SYSTEM:
                continue
            # Skip critical priority messages
            if message.priority == Priority.CRITICAL:
                continue
            # This message is eligible for summarization
            summarizable.append(message)
        # Return the filtered list of summarizable messages
        return summarizable

    def _replace_with_summary(self, summarized: list[Message], summary: Message) -> None:
        """Replace summarized messages with the summary message in the state.

        Removes the original messages and inserts the summary in their place,
        preserving the position relative to other messages.

        Args:
            summarized: The messages that were summarized.
            summary: The condensed summary message to insert.
        """
        # Create a set of summarized message IDs for fast lookup
        summarized_set = {id(msg) for msg in summarized}
        # Find the index of the first summarized message
        insert_idx = 0
        # Search for the first summarized message's position
        for idx, msg in enumerate(self._state.messages):
            if id(msg) in summarized_set:
                insert_idx = idx
                break
        # Remove all summarized messages from the state
        self._state.messages = [
            msg for msg in self._state.messages if id(msg) not in summarized_set
        ]
        # Insert the summary message at the original position
        self._state.messages.insert(insert_idx, summary)

    def _validate_message(self, message: Message) -> Message:
        """Validate a message against configured security constraints.

        Checks message length, sanitizes content if configured, and
        verifies the conversation hasn't exceeded the message limit.

        Args:
            message: The message to validate.

        Returns:
            The validated (and possibly sanitized) message.

        Raises:
            ValidationError: If the message fails validation.
        """
        # Check that the message content doesn't exceed the length limit
        if len(message.content) > self._config.security.max_message_length:
            raise ValidationError(
                f"Message content exceeds maximum length: "
                f"{len(message.content)} > {self._config.security.max_message_length}"
            )
        # Check that the conversation hasn't exceeded the message count limit
        if len(self._state.messages) >= self._config.security.max_messages:
            raise ValidationError(
                f"Maximum message count reached: {self._config.security.max_messages}"
            )
        # Sanitize the content if configured to strip control characters
        if self._config.security.sanitize_input:
            # Remove control characters but preserve newlines and tabs
            sanitized_content = CONTROL_CHAR_PATTERN.sub("", message.content)
            # Only create a new message if content was actually modified
            if sanitized_content != message.content:
                # Log the sanitization at debug level
                logger.debug("Sanitized control characters from message content")
                # Return a new message with the sanitized content
                return Message(
                    role=message.role,
                    content=sanitized_content,
                    priority=message.priority,
                    timestamp=message.timestamp,
                    token_count=message.token_count,
                    metadata=message.metadata,
                )
        # Return the original message if no sanitization was needed
        return message

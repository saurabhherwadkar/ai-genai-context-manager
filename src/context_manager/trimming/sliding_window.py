"""Sliding window trimming strategy.

Keeps only the N most recent messages, always preserving the system
message at the start of the conversation for context continuity.
"""

import logging

from context_manager.models import Message, Priority, Role

# Module-level logger for trimming diagnostics
logger = logging.getLogger(__name__)


class SlidingWindowStrategy:
    """Trims messages by keeping only a fixed window of recent messages.

    Maintains a sliding window of the most recent N messages while
    optionally preserving the system message regardless of window position.
    """

    def __init__(
        self,
        window_size: int = 50,
        preserve_system_message: bool = True,
    ) -> None:
        """Initialize the sliding window strategy.

        Args:
            window_size: Maximum number of recent messages to retain.
            preserve_system_message: If True, always keep system messages.
        """
        # Store the maximum window size for message retention
        self._window_size = window_size
        # Store whether system messages bypass the window constraint
        self._preserve_system = preserve_system_message
        # Log the strategy configuration
        logger.debug(
            "SlidingWindowStrategy initialized (window=%d, preserve_system=%s)",
            window_size,
            preserve_system_message,
        )

    def trim(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counts: list[int],
    ) -> list[Message]:
        """Trim messages to keep only the most recent within the window.

        First applies the window size constraint, then additionally
        trims if the result still exceeds the token budget.

        Args:
            messages: The full list of messages to trim.
            max_tokens: The maximum allowed total token count.
            token_counts: Pre-computed token count for each message.

        Returns:
            A new list containing only the retained messages.
        """
        # If messages fit in the window and within budget, return unchanged
        if len(messages) <= self._window_size and sum(token_counts) <= max_tokens:
            # Log that no trimming was needed
            logger.debug("No trimming needed: %d messages within window", len(messages))
            # Return a shallow copy to avoid mutating the original
            return list(messages)
        # Separate system messages from non-system messages
        system_messages: list[tuple[int, Message]] = []
        # Collect non-system messages with their original indices
        non_system_messages: list[tuple[int, Message]] = []
        # Partition messages based on their role
        for idx, message in enumerate(messages):
            # System messages go to the preserved list if configured
            if self._preserve_system and message.role == Role.SYSTEM:
                system_messages.append((idx, message))
            else:
                # Non-system messages are candidates for windowing
                non_system_messages.append((idx, message))
        # Calculate how many non-system messages the window allows
        available_slots = self._window_size - len(system_messages)
        # Ensure at least 1 slot remains for conversation content
        available_slots = max(1, available_slots)
        # Take only the most recent messages within the window limit
        windowed = non_system_messages[-available_slots:]
        # Combine system messages and windowed messages
        retained_indices = {idx for idx, _ in system_messages} | {idx for idx, _ in windowed}
        # Build the result preserving original message order
        result: list[Message] = []
        # Collect token counts for retained messages
        result_tokens: list[int] = []
        # Iterate through original messages keeping only retained ones
        for idx, message in enumerate(messages):
            # Include this message if it's in the retained set
            if idx in retained_indices:
                result.append(message)
                result_tokens.append(token_counts[idx])
        # Apply token budget constraint to the windowed result
        current_total = sum(result_tokens)
        # Further trim from oldest if still over token budget
        if current_total > max_tokens:
            # Apply secondary FIFO trim to the windowed results
            result = self._trim_to_budget(result, result_tokens, max_tokens)
        # Log the trimming summary
        logger.info(
            "Sliding window retained %d of %d messages",
            len(result),
            len(messages),
        )
        # Return the final trimmed message list
        return result

    def _trim_to_budget(
        self,
        messages: list[Message],
        token_counts: list[int],
        max_tokens: int,
    ) -> list[Message]:
        """Further trim windowed messages to meet the token budget.

        Applies FIFO removal within the window when token budget is
        exceeded even after windowing.

        Args:
            messages: Pre-windowed messages that exceed the budget.
            token_counts: Token counts for each message.
            max_tokens: The maximum allowed total.

        Returns:
            Messages trimmed to fit within the token budget.
        """
        # Calculate how much we need to reduce
        current_total = sum(token_counts)
        # Track indices to remove from oldest within the window
        indices_to_remove: set[int] = set()
        # Iterate from oldest to newest within the window
        for idx in range(len(messages)):
            # Stop once we're within the budget
            if current_total <= max_tokens:
                break
            # Never remove critical priority messages
            if messages[idx].priority == Priority.CRITICAL:
                continue
            # Never remove system messages if preservation is on
            if self._preserve_system and messages[idx].role == Role.SYSTEM:
                continue
            # Mark for removal and reduce the running total
            indices_to_remove.add(idx)
            # Subtract this message's tokens from the total
            current_total -= token_counts[idx]
        # Build and return the filtered result list
        return [msg for idx, msg in enumerate(messages) if idx not in indices_to_remove]

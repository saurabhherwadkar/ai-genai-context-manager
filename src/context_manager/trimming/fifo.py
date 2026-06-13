"""First-In-First-Out (FIFO) trimming strategy.

Removes the oldest messages first when the conversation exceeds
the token budget. Optionally preserves the system message.
"""

import logging

from context_manager.models import Message, Priority, Role

# Module-level logger for trimming diagnostics
logger = logging.getLogger(__name__)


class FifoTrimmingStrategy:
    """Trims messages in chronological order, removing oldest first.

    The simplest trimming strategy: removes messages from the beginning
    of the conversation until the token budget is satisfied.
    """

    def __init__(self, preserve_system_message: bool = True) -> None:
        """Initialize the FIFO trimming strategy.

        Args:
            preserve_system_message: If True, never trim system messages.
        """
        # Store whether system messages should be protected from trimming
        self._preserve_system = preserve_system_message
        # Log the strategy configuration at debug level
        logger.debug(
            "FifoTrimmingStrategy initialized (preserve_system=%s)",
            preserve_system_message,
        )

    def trim(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counts: list[int],
    ) -> list[Message]:
        """Trim messages by removing the oldest non-protected messages first.

        Args:
            messages: The full list of messages to trim.
            max_tokens: The maximum allowed total token count.
            token_counts: Pre-computed token count for each message.

        Returns:
            A new list containing only the retained messages.
        """
        # Calculate the current total token count across all messages
        current_total = sum(token_counts)
        # If already within budget, return all messages unchanged
        if current_total <= max_tokens:
            # Log that no trimming was needed
            logger.debug("No trimming needed: %d <= %d tokens", current_total, max_tokens)
            # Return a shallow copy to avoid mutating the original list
            return list(messages)
        # Track which message indices to remove (oldest first)
        indices_to_remove: set[int] = set()
        # Iterate from oldest to newest looking for messages to remove
        for idx in range(len(messages)):
            # Stop removing once we're within the token budget
            if current_total <= max_tokens:
                break
            # Skip system messages if preservation is enabled
            if self._preserve_system and messages[idx].role == Role.SYSTEM:
                continue
            # Skip critical priority messages as they must never be trimmed
            if messages[idx].priority == Priority.CRITICAL:
                continue
            # Mark this message for removal and subtract its tokens
            indices_to_remove.add(idx)
            # Reduce the running total by this message's token count
            current_total -= token_counts[idx]
        # Build the result list excluding removed message indices
        result = [msg for idx, msg in enumerate(messages) if idx not in indices_to_remove]
        # Log the trimming result for diagnostics
        logger.info(
            "FIFO trimmed %d messages, %d remaining (tokens: %d)",
            len(indices_to_remove),
            len(result),
            current_total,
        )
        # Return the trimmed message list
        return result

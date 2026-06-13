"""Priority-based trimming strategy.

Removes messages based on their priority level, trimming lowest
priority messages first regardless of their position in the conversation.
"""

import logging

from context_manager.models import Message, Priority, Role

# Module-level logger for trimming diagnostics
logger = logging.getLogger(__name__)

# Define the trimming order: low priority messages are removed first
PRIORITY_TRIM_ORDER: list[Priority] = [
    Priority.LOW,
    Priority.NORMAL,
    Priority.HIGH,
]


class PriorityTrimmingStrategy:
    """Trims messages based on priority level, lowest first.

    Processes messages in priority tiers: all LOW messages are trimmed
    before any NORMAL messages, and all NORMAL before any HIGH.
    CRITICAL messages are never trimmed.
    """

    def __init__(self, preserve_system_message: bool = True) -> None:
        """Initialize the priority trimming strategy.

        Args:
            preserve_system_message: If True, never trim system messages.
        """
        # Store whether system messages should be immune to trimming
        self._preserve_system = preserve_system_message
        # Log the strategy configuration
        logger.debug(
            "PriorityTrimmingStrategy initialized (preserve_system=%s)",
            preserve_system_message,
        )

    def trim(
        self,
        messages: list[Message],
        max_tokens: int,
        token_counts: list[int],
    ) -> list[Message]:
        """Trim messages by removing lowest priority messages first.

        Processes priority tiers in order (LOW, NORMAL, HIGH), removing
        oldest messages within each tier first. CRITICAL messages and
        optionally system messages are never removed.

        Args:
            messages: The full list of messages to trim.
            max_tokens: The maximum allowed total token count.
            token_counts: Pre-computed token count for each message.

        Returns:
            A new list containing only the retained messages.
        """
        # Calculate the current total token usage
        current_total = sum(token_counts)
        # If already within budget, return all messages unchanged
        if current_total <= max_tokens:
            # Log that no trimming was necessary
            logger.debug("No trimming needed: %d <= %d tokens", current_total, max_tokens)
            # Return a shallow copy to preserve the original list
            return list(messages)
        # Track which message indices will be removed
        indices_to_remove: set[int] = set()
        # Process each priority tier from lowest to highest
        for priority_level in PRIORITY_TRIM_ORDER:
            # Stop if we've reduced enough to meet the budget
            if current_total <= max_tokens:
                break
            # Find all messages at this priority level (oldest first)
            for idx in range(len(messages)):
                # Stop if budget is now satisfied
                if current_total <= max_tokens:
                    break
                # Skip messages already marked for removal
                if idx in indices_to_remove:
                    continue
                # Skip messages not at the current priority tier
                if messages[idx].priority != priority_level:
                    continue
                # Skip system messages if preservation is enabled
                if self._preserve_system and messages[idx].role == Role.SYSTEM:
                    continue
                # Mark this message for removal
                indices_to_remove.add(idx)
                # Subtract its token count from the running total
                current_total -= token_counts[idx]
        # Build the result list excluding all removed indices
        result = [msg for idx, msg in enumerate(messages) if idx not in indices_to_remove]
        # Log the trimming summary with priority breakdown
        logger.info(
            "Priority trimmed %d messages, %d remaining (tokens: %d)",
            len(indices_to_remove),
            len(result),
            current_total,
        )
        # Return the trimmed message list
        return result

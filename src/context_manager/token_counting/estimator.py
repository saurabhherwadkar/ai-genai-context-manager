"""Character-ratio based token estimator for offline use.

Provides a fast, zero-dependency fallback for token estimation
when exact counting via tiktoken or API is not needed or available.
"""

import logging
import math
from collections.abc import Sequence

from context_manager.models import Message, TokenCount

# Module-level logger for estimator diagnostics
logger = logging.getLogger(__name__)


class EstimatorTokenCounter:
    """Token counter using character-to-token ratio estimation.

    Provides approximate token counts without external dependencies
    or API calls. Useful as a fallback or for quick estimates.
    """

    def __init__(self, ratio: float = 4.0) -> None:
        """Initialize the estimator with a characters-per-token ratio.

        The default ratio of 4.0 is a reasonable approximation for
        English text across most LLM tokenizers.

        Args:
            ratio: Average number of characters per token.
        """
        # Validate that the ratio is positive to avoid division errors
        if ratio <= 0:
            raise ValueError("Estimator ratio must be a positive number")
        # Store the characters-per-token ratio for calculations
        self._ratio = ratio
        # Log initialization with the configured ratio
        logger.debug("EstimatorTokenCounter initialized with ratio: %.2f", ratio)

    async def count_tokens(self, messages: Sequence[Message]) -> TokenCount:
        """Estimate token counts for a sequence of messages.

        Uses the character length divided by the ratio, rounding up
        to avoid underestimation of token budgets.

        Args:
            messages: Sequence of Message objects to estimate.

        Returns:
            TokenCount with estimated total and per-message breakdown.
        """
        # Build per-message estimates using the character ratio
        per_message_counts: list[int] = []
        # Iterate through each message to estimate its token count
        for message in messages:
            # Estimate tokens by dividing character count by the ratio
            estimated = self._estimate_text(message.content)
            # Add overhead for the role field (approximately 1 token)
            estimated += 1
            # Append the per-message estimate to the results list
            per_message_counts.append(estimated)
        # Sum all per-message estimates for the total
        total = sum(per_message_counts)
        # Log the estimation result at debug level
        logger.debug("Estimated %d total tokens for %d messages", total, len(messages))
        # Return the structured token count with estimates
        return TokenCount(total=total, per_message=per_message_counts)

    async def count_single(self, text: str) -> int:
        """Estimate token count for a single text string.

        Args:
            text: The text string to estimate tokens for.

        Returns:
            The estimated number of tokens.
        """
        # Delegate to the internal estimation method
        return self._estimate_text(text)

    def _estimate_text(self, text: str) -> int:
        """Estimate tokens for a text string using the character ratio.

        Rounds up to avoid underestimation which could cause
        unexpected context window overflow.

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count, minimum of 1 for non-empty text.
        """
        # Return 0 for empty strings since they contain no tokens
        if not text:
            return 0
        # Divide character count by ratio and round up to be conservative
        return max(1, math.ceil(len(text) / self._ratio))

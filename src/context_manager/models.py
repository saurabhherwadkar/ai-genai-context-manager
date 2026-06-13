"""Core data models for the context manager package.

These Pydantic models define the structure of messages, token counts,
and conversation state used throughout the system.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Enumeration of valid message roles in a conversation."""

    # System instructions that guide the assistant's behavior
    SYSTEM = "system"
    # Messages from the human user
    USER = "user"
    # Messages from the AI assistant
    ASSISTANT = "assistant"


class Priority(StrEnum):
    """Priority levels that control trimming order for messages.

    Higher priority messages are preserved longer during trimming.
    """

    # Never trimmed under any circumstances (e.g., system prompts)
    CRITICAL = "critical"
    # Trimmed only after all normal and low priority messages
    HIGH = "high"
    # Default priority level for regular conversation messages
    NORMAL = "normal"
    # Trimmed first when token budget is exceeded
    LOW = "low"


class Message(BaseModel, frozen=True):
    """Immutable representation of a single conversation message.

    Frozen to ensure messages can be safely shared without copying.
    The token_count field is optionally populated by token counters.
    """

    # The role of the entity that produced this message
    role: Role
    # The text content of the message
    content: str
    # Priority level controlling trim order (defaults to normal)
    priority: Priority = Priority.NORMAL
    # UTC timestamp when the message was created
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Cached token count populated lazily by token counters
    token_count: int | None = None
    # Arbitrary key-value metadata for extension and tracking
    metadata: dict[str, str] = Field(default_factory=dict)


class TokenCount(BaseModel, frozen=True):
    """Immutable container for token count results.

    Holds both the aggregate total and per-message breakdown.
    """

    # Sum of all individual message token counts
    total: int
    # Token count for each message in order
    per_message: list[int]


class ConversationState(BaseModel):
    """Mutable state tracking the current conversation context window.

    Updated by the ContextManager as messages are added or trimmed.
    """

    # Ordered list of messages in the current context window
    messages: list[Message] = Field(default_factory=list)
    # Current total token count across all messages
    total_tokens: int = 0
    # Maximum allowed tokens for the context window
    max_tokens: int
    # Tokens reserved for the system prompt and response generation
    reserved_tokens: int = 0
    # Number of messages that have been trimmed from this conversation
    trimmed_count: int = 0
    # Number of messages that were summarized into a condensed form
    summarized_count: int = 0

    @property
    def available_tokens(self) -> int:
        """Calculate the remaining token budget for new messages."""
        # Subtract reserved tokens and current usage from the maximum
        return self.max_tokens - self.reserved_tokens - self.total_tokens

    @property
    def message_count(self) -> int:
        """Return the number of messages currently in the context window."""
        # Simply delegate to the length of the messages list
        return len(self.messages)

    @property
    def is_over_budget(self) -> bool:
        """Check whether the current token usage exceeds the budget."""
        # Compare current total against the effective budget limit
        return self.total_tokens > (self.max_tokens - self.reserved_tokens)

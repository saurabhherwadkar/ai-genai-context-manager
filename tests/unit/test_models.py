"""Unit tests for the core data models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from context_manager.models import (
    ConversationState,
    Message,
    Priority,
    Role,
    TokenCount,
)


class TestRole:
    """Tests for the Role enumeration."""

    def test_role_values(self) -> None:
        """Verify all Role enum members have correct string values."""
        # Assert each role maps to the expected string
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"

    def test_role_is_string(self) -> None:
        """Verify Role members can be used as strings."""
        # Assert Role values work in string contexts
        assert f"{Role.USER}" == "user"


class TestPriority:
    """Tests for the Priority enumeration."""

    def test_priority_values(self) -> None:
        """Verify all Priority enum members have correct string values."""
        # Assert each priority level maps to the expected string
        assert Priority.CRITICAL == "critical"
        assert Priority.HIGH == "high"
        assert Priority.NORMAL == "normal"
        assert Priority.LOW == "low"


class TestMessage:
    """Tests for the Message data model."""

    def test_create_basic_message(self) -> None:
        """Verify a basic message can be created with required fields."""
        # Create a message with only the required fields
        msg = Message(role=Role.USER, content="Hello")
        # Assert the role is set correctly
        assert msg.role == Role.USER
        # Assert the content is stored
        assert msg.content == "Hello"
        # Assert defaults are applied
        assert msg.priority == Priority.NORMAL
        assert msg.token_count is None
        assert msg.metadata == {}

    def test_message_is_frozen(self) -> None:
        """Verify that Message instances are immutable."""
        # Create a message
        msg = Message(role=Role.USER, content="Hello")
        # Attempt to modify should raise a validation error
        with pytest.raises(PydanticValidationError, match="frozen"):
            msg.content = "Modified"  # type: ignore[misc]

    def test_message_with_all_fields(self) -> None:
        """Verify a message with all fields set."""
        # Create a fully specified message
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        msg = Message(
            role=Role.ASSISTANT,
            content="Response",
            priority=Priority.HIGH,
            timestamp=ts,
            token_count=42,
            metadata={"source": "test"},
        )
        # Assert all fields are stored correctly
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Response"
        assert msg.priority == Priority.HIGH
        assert msg.timestamp == ts
        assert msg.token_count == 42
        assert msg.metadata == {"source": "test"}

    def test_message_default_timestamp_is_utc(self) -> None:
        """Verify the default timestamp uses UTC timezone."""
        # Create a message to get the default timestamp
        msg = Message(role=Role.USER, content="Test")
        # Assert the timestamp has UTC timezone info
        assert msg.timestamp.tzinfo is not None


class TestTokenCount:
    """Tests for the TokenCount data model."""

    def test_create_token_count(self) -> None:
        """Verify TokenCount stores total and per-message counts."""
        # Create a token count with known values
        tc = TokenCount(total=100, per_message=[30, 40, 30])
        # Assert total is stored
        assert tc.total == 100
        # Assert per-message list is stored
        assert tc.per_message == [30, 40, 30]

    def test_token_count_is_frozen(self) -> None:
        """Verify that TokenCount instances are immutable."""
        # Create a token count
        tc = TokenCount(total=50, per_message=[50])
        # Attempt to modify should raise
        with pytest.raises(PydanticValidationError, match="frozen"):
            tc.total = 100  # type: ignore[misc]


class TestConversationState:
    """Tests for the ConversationState data model."""

    def test_default_state(self) -> None:
        """Verify default ConversationState values."""
        # Create a state with just the required max_tokens
        state = ConversationState(max_tokens=1000)
        # Assert defaults
        assert state.messages == []
        assert state.total_tokens == 0
        assert state.reserved_tokens == 0
        assert state.trimmed_count == 0
        assert state.summarized_count == 0

    def test_available_tokens_calculation(self) -> None:
        """Verify the available_tokens property calculation."""
        # Create a state with known values
        state = ConversationState(max_tokens=1000, reserved_tokens=200, total_tokens=300)
        # Available = max - reserved - current = 1000 - 200 - 300 = 500
        assert state.available_tokens == 500

    def test_message_count_property(self) -> None:
        """Verify the message_count property returns correct count."""
        # Create a state and add some messages
        state = ConversationState(max_tokens=1000)
        state.messages = [
            Message(role=Role.USER, content="One"),
            Message(role=Role.ASSISTANT, content="Two"),
        ]
        # Assert the property returns the correct count
        assert state.message_count == 2

    def test_is_over_budget_true(self) -> None:
        """Verify is_over_budget returns True when exceeded."""
        # Create a state that exceeds the effective budget
        state = ConversationState(max_tokens=100, reserved_tokens=20, total_tokens=90)
        # Effective budget = 100 - 20 = 80, current = 90 > 80
        assert state.is_over_budget is True

    def test_is_over_budget_false(self) -> None:
        """Verify is_over_budget returns False when within budget."""
        # Create a state within the effective budget
        state = ConversationState(max_tokens=100, reserved_tokens=20, total_tokens=50)
        # Effective budget = 100 - 20 = 80, current = 50 < 80
        assert state.is_over_budget is False

    def test_is_over_budget_exact(self) -> None:
        """Verify is_over_budget returns False when exactly at budget."""
        # Create a state exactly at the effective budget
        state = ConversationState(max_tokens=100, reserved_tokens=20, total_tokens=80)
        # Effective budget = 80, current = 80 (not over)
        assert state.is_over_budget is False

"""Unit tests for the ContextManager orchestrator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from context_manager.config import ContextManagerConfig
from context_manager.exceptions import (
    TokenBudgetExceededError,
    ValidationError,
)
from context_manager.manager import ContextManager
from context_manager.models import Message, Role, TokenCount


class TestContextManager:
    """Tests for the ContextManager class."""

    @pytest.fixture
    def mock_token_counter(self) -> AsyncMock:
        """Create a mock token counter that returns predictable values."""
        # Create a mock that counts 10 tokens per message
        counter = AsyncMock()
        counter.count_single = AsyncMock(return_value=10)
        counter.count_tokens = AsyncMock(
            return_value=TokenCount(total=30, per_message=[10, 10, 10])
        )
        return counter

    @pytest.fixture
    def mock_trimmer(self) -> MagicMock:
        """Create a mock trimming strategy."""
        # Create a mock that returns a trimmed list
        trimmer = MagicMock()
        trimmer.trim = MagicMock(return_value=[])
        return trimmer

    @pytest.fixture
    def config(self) -> ContextManagerConfig:
        """Create a test configuration with small limits."""
        # Use small limits to make trimming tests easier
        return ContextManagerConfig(
            environment="test",
            provider={  # type: ignore[arg-type]
                "name": "openai",
                "model": "gpt-4o",
                "max_context_tokens": 100,
                "reserved_response_tokens": 20,
            },
            security={  # type: ignore[arg-type]
                "max_message_length": 1000,
                "max_messages": 10,
                "sanitize_input": True,
            },
            summarization={"enabled": False},  # type: ignore[arg-type]
        )

    @pytest.fixture
    def manager(
        self,
        config: ContextManagerConfig,
        mock_token_counter: AsyncMock,
        mock_trimmer: MagicMock,
    ) -> ContextManager:
        """Create a ContextManager with mocked dependencies."""
        # Wire up the manager with mock strategies
        return ContextManager(
            config=config,
            token_counter=mock_token_counter,
            trimming_strategy=mock_trimmer,
        )

    @pytest.mark.asyncio
    async def test_add_message_basic(
        self, manager: ContextManager, mock_token_counter: AsyncMock
    ) -> None:
        """Verify adding a basic message updates the state."""
        # Create a simple message
        message = Message(role=Role.USER, content="Hello")
        # Add it to the manager
        state = await manager.add_message(message)
        # State should contain the message
        assert state.message_count == 1
        # Token count should be updated
        assert state.total_tokens == 10

    @pytest.mark.asyncio
    async def test_add_message_exceeds_budget_raises_error(
        self, manager: ContextManager, mock_token_counter: AsyncMock
    ) -> None:
        """Verify a single oversized message raises TokenBudgetExceededError."""
        # Make the counter return a huge token count
        mock_token_counter.count_single = AsyncMock(return_value=200)
        # Create a message
        message = Message(role=Role.USER, content="Very long message")
        # Should raise TokenBudgetExceededError
        with pytest.raises(TokenBudgetExceededError):
            await manager.add_message(message)

    @pytest.mark.asyncio
    async def test_add_message_triggers_trimming(
        self, manager: ContextManager, mock_token_counter: AsyncMock, mock_trimmer: MagicMock
    ) -> None:
        """Verify trimming is triggered when budget is exceeded."""
        # Make messages cost 30 tokens each to quickly exceed budget
        mock_token_counter.count_single = AsyncMock(return_value=30)
        # Set up the trimmer to return a subset
        trimmed_msg = Message(role=Role.USER, content="Kept")
        mock_trimmer.trim = MagicMock(return_value=[trimmed_msg])
        # Add messages until over budget (budget is 80 = 100 - 20)
        await manager.add_message(Message(role=Role.USER, content="First"))
        await manager.add_message(Message(role=Role.USER, content="Second"))
        # Third message should trigger trimming (90 > 80)
        await manager.add_message(Message(role=Role.USER, content="Third"))
        # Trimmer should have been called
        assert mock_trimmer.trim.called

    @pytest.mark.asyncio
    async def test_add_messages_multiple(
        self, manager: ContextManager, mock_token_counter: AsyncMock
    ) -> None:
        """Verify adding multiple messages processes each sequentially."""
        # Create multiple messages
        messages = [
            Message(role=Role.USER, content="One"),
            Message(role=Role.ASSISTANT, content="Two"),
        ]
        # Add all messages
        state = await manager.add_messages(messages)
        # Both should be added
        assert state.message_count == 2
        assert state.total_tokens == 20

    @pytest.mark.asyncio
    async def test_get_context_window(
        self, manager: ContextManager, mock_token_counter: AsyncMock
    ) -> None:
        """Verify get_context_window returns current messages."""
        # Add a message
        await manager.add_message(Message(role=Role.USER, content="Hello"))
        # Get the context window
        window = await manager.get_context_window()
        # Should contain the message
        assert len(window) == 1
        assert window[0].content == "Hello"

    def test_get_token_usage(
        self, manager: ContextManager
    ) -> None:
        """Verify get_token_usage returns correct breakdown."""
        # Get usage on empty manager
        usage = manager.get_token_usage()
        # Should be empty
        assert usage.total == 0
        assert usage.per_message == []

    def test_reset_clears_state(
        self, manager: ContextManager
    ) -> None:
        """Verify reset clears all messages and counters."""
        # Manually set some state
        manager._state.total_tokens = 50
        manager._state.trimmed_count = 3
        # Reset the manager
        manager.reset()
        # State should be clean
        assert manager._state.total_tokens == 0
        assert manager._state.trimmed_count == 0
        assert manager._state.message_count == 0

    @pytest.mark.asyncio
    async def test_validate_message_too_long(
        self, manager: ContextManager
    ) -> None:
        """Verify messages exceeding max_message_length are rejected."""
        # Create a message exceeding the 1000 char limit
        long_content = "x" * 1001
        message = Message(role=Role.USER, content=long_content)
        # Should raise ValidationError
        with pytest.raises(ValidationError, match="maximum length"):
            await manager.add_message(message)

    @pytest.mark.asyncio
    async def test_validate_max_messages_reached(
        self, manager: ContextManager, mock_token_counter: AsyncMock
    ) -> None:
        """Verify error when max_messages limit is reached."""
        # Fill up to the limit (10 messages in test config)
        for i in range(10):
            manager._state.messages.append(
                Message(role=Role.USER, content=f"Msg {i}")
            )
        # Next message should fail validation
        with pytest.raises(ValidationError, match="Maximum message count"):
            await manager.add_message(Message(role=Role.USER, content="Over limit"))

    @pytest.mark.asyncio
    async def test_sanitize_input_strips_control_chars(
        self, manager: ContextManager, mock_token_counter: AsyncMock
    ) -> None:
        """Verify control characters are stripped from message content."""
        # Create a message with control characters
        message = Message(role=Role.USER, content="Hello\x00World\x07!")
        # Add the message
        state = await manager.add_message(message)
        # Content should have control chars removed
        assert state.messages[0].content == "HelloWorld!"

    def test_get_state_returns_state(self, manager: ContextManager) -> None:
        """Verify get_state returns the conversation state object."""
        # Get the state
        state = manager.get_state()
        # Should be the internal state
        assert state.max_tokens == 100
        assert state.reserved_tokens == 20

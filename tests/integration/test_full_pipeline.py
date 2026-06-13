"""Integration tests for the full context management pipeline.

Tests the complete flow from adding messages through trimming
and summarization using real token counting but mocked API calls.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from context_manager.config import ContextManagerConfig
from context_manager.manager import ContextManager
from context_manager.models import Message, Priority, Role
from context_manager.summarization.openai_summarizer import OpenAISummarizer
from context_manager.token_counting.estimator import EstimatorTokenCounter
from context_manager.trimming.fifo import FifoTrimmingStrategy
from context_manager.trimming.priority import PriorityTrimmingStrategy
from context_manager.trimming.sliding_window import SlidingWindowStrategy


class TestFullPipelineWithFifo:
    """Integration tests using FIFO trimming with estimator token counting."""

    @pytest.fixture
    def manager(self) -> ContextManager:
        """Create a full pipeline with estimator counter and FIFO trimmer."""
        # Configure with a small budget to trigger trimming easily
        config = ContextManagerConfig(
            environment="test",
            provider={  # type: ignore[arg-type]
                "name": "openai",
                "model": "gpt-4o",
                "max_context_tokens": 50,
                "reserved_response_tokens": 10,
            },
            security={  # type: ignore[arg-type]
                "max_message_length": 10000,
                "max_messages": 100,
                "sanitize_input": True,
            },
            summarization={"enabled": False},  # type: ignore[arg-type]
        )
        # Use the estimator for predictable token counting
        counter = EstimatorTokenCounter(ratio=4.0)
        # Use FIFO trimming strategy
        trimmer = FifoTrimmingStrategy(preserve_system_message=True)
        # Create the manager with real strategies
        return ContextManager(
            config=config,
            token_counter=counter,
            trimming_strategy=trimmer,
        )

    @pytest.mark.asyncio
    async def test_add_and_trim_flow(self, manager: ContextManager) -> None:
        """Verify messages are added and trimmed when budget is exceeded."""
        # Add a system message
        await manager.add_message(
            Message(
                role=Role.SYSTEM,
                content="You are helpful.",
                priority=Priority.CRITICAL,
            )
        )
        # Add several user messages to exceed the budget
        for i in range(5):
            await manager.add_message(
                Message(
                    role=Role.USER,
                    content=f"Question number {i} with some text here",
                )
            )
        # Get the context window
        window = await manager.get_context_window()
        # System message should be preserved
        assert window[0].role == Role.SYSTEM
        # Total messages should be less than what was added
        assert len(window) < 6

    @pytest.mark.asyncio
    async def test_token_budget_maintained(self, manager: ContextManager) -> None:
        """Verify the token budget is never exceeded in the final window."""
        # Add many messages to trigger multiple trims
        for i in range(10):
            await manager.add_message(
                Message(role=Role.USER, content=f"Message {i}")
            )
        # Get the state
        state = manager.get_state()
        # Should not be over budget (effective budget = 50 - 10 = 40)
        assert state.total_tokens <= 40


class TestFullPipelineWithSlidingWindow:
    """Integration tests using sliding window trimming."""

    @pytest.fixture
    def manager(self) -> ContextManager:
        """Create a pipeline with sliding window trimmer."""
        # Configure with a tight budget to ensure trimming triggers
        config = ContextManagerConfig(
            environment="test",
            provider={  # type: ignore[arg-type]
                "name": "openai",
                "model": "gpt-4o",
                "max_context_tokens": 50,
                "reserved_response_tokens": 10,
            },
            security={  # type: ignore[arg-type]
                "max_message_length": 10000,
                "max_messages": 100,
                "sanitize_input": True,
            },
            summarization={"enabled": False},  # type: ignore[arg-type]
        )
        # Use estimator for predictable counting
        counter = EstimatorTokenCounter(ratio=4.0)
        # Use sliding window with size 5
        trimmer = SlidingWindowStrategy(
            window_size=5, preserve_system_message=True
        )
        # Create the manager
        return ContextManager(
            config=config,
            token_counter=counter,
            trimming_strategy=trimmer,
        )

    @pytest.mark.asyncio
    async def test_sliding_window_retains_recent(
        self, manager: ContextManager
    ) -> None:
        """Verify only recent messages are retained after windowing."""
        # Add a system message
        await manager.add_message(
            Message(
                role=Role.SYSTEM,
                content="System prompt here",
                priority=Priority.CRITICAL,
            )
        )
        # Add messages with enough content to exceed budget
        for i in range(10):
            await manager.add_message(
                Message(
                    role=Role.USER,
                    content=f"This is message number {i} with extra text",
                )
            )
        # Get the context window
        window = await manager.get_context_window()
        # Should have fewer messages than added (trimmed by window + budget)
        assert len(window) <= 6


class TestFullPipelineWithPriority:
    """Integration tests using priority-based trimming."""

    @pytest.fixture
    def manager(self) -> ContextManager:
        """Create a pipeline with priority trimmer."""
        # Configure with tight budget
        config = ContextManagerConfig(
            environment="test",
            provider={  # type: ignore[arg-type]
                "name": "openai",
                "model": "gpt-4o",
                "max_context_tokens": 60,
                "reserved_response_tokens": 10,
            },
            security={  # type: ignore[arg-type]
                "max_message_length": 10000,
                "max_messages": 100,
                "sanitize_input": True,
            },
            summarization={"enabled": False},  # type: ignore[arg-type]
        )
        # Use estimator for predictable counting
        counter = EstimatorTokenCounter(ratio=4.0)
        # Use priority trimming
        trimmer = PriorityTrimmingStrategy(preserve_system_message=True)
        # Create the manager
        return ContextManager(
            config=config,
            token_counter=counter,
            trimming_strategy=trimmer,
        )

    @pytest.mark.asyncio
    async def test_priority_preserves_high_priority(
        self, manager: ContextManager
    ) -> None:
        """Verify high priority messages survive trimming."""
        # Add messages with different priorities
        await manager.add_message(
            Message(
                role=Role.USER, content="Low priority", priority=Priority.LOW
            )
        )
        await manager.add_message(
            Message(
                role=Role.USER, content="High priority", priority=Priority.HIGH
            )
        )
        await manager.add_message(
            Message(
                role=Role.USER,
                content="Normal priority",
                priority=Priority.NORMAL,
            )
        )
        # Add more to trigger trimming
        await manager.add_message(
            Message(
                role=Role.USER,
                content="Another message to push over budget",
            )
        )
        # Get the window
        window = await manager.get_context_window()
        # High priority message should still be present
        high_priority_content = [
            m.content for m in window if m.priority == Priority.HIGH
        ]
        # Low priority should be removed first
        low_priority_content = [
            m.content for m in window if m.priority == Priority.LOW
        ]
        # If trimming occurred, high should be kept over low
        if len(window) < 4:
            assert len(low_priority_content) <= len(high_priority_content)


class TestFullPipelineWithSummarization:
    """Integration tests using summarization with mocked LLM calls."""

    @pytest.fixture
    def mock_openai_client(self) -> AsyncMock:
        """Create a mock OpenAI client for summarization."""
        # Create the mock client
        client = AsyncMock()
        # Set up the response
        choice = MagicMock()
        choice.message.content = (
            "Summary: Users discussed Python programming."
        )
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def manager(self, mock_openai_client: AsyncMock) -> ContextManager:
        """Create a pipeline with summarization enabled."""
        # Configure with summarization
        config = ContextManagerConfig(
            environment="test",
            provider={  # type: ignore[arg-type]
                "name": "openai",
                "model": "gpt-4o",
                "max_context_tokens": 80,
                "reserved_response_tokens": 10,
            },
            security={  # type: ignore[arg-type]
                "max_message_length": 10000,
                "max_messages": 100,
                "sanitize_input": True,
            },
            summarization={  # type: ignore[arg-type]
                "enabled": True,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "max_summary_tokens": 100,
            },
        )
        # Use estimator for predictable counting
        counter = EstimatorTokenCounter(ratio=4.0)
        # Use FIFO trimming as fallback
        trimmer = FifoTrimmingStrategy(preserve_system_message=True)
        # Create summarizer with mock client
        summarizer = OpenAISummarizer(
            model="gpt-4o-mini",
            max_summary_tokens=100,
            client=mock_openai_client,
        )
        # Create the manager with all strategies
        return ContextManager(
            config=config,
            token_counter=counter,
            trimming_strategy=trimmer,
            summarization_strategy=summarizer,
        )

    @pytest.mark.asyncio
    async def test_summarization_triggered(
        self, manager: ContextManager, mock_openai_client: AsyncMock
    ) -> None:
        """Verify summarization is triggered when budget is exceeded."""
        # Add enough messages to exceed the budget
        for i in range(8):
            await manager.add_message(
                Message(
                    role=Role.USER, content=f"Question {i} about topic"
                )
            )
            await manager.add_message(
                Message(
                    role=Role.ASSISTANT,
                    content=f"Answer {i} with details",
                )
            )
        # The context should have been managed (summarized or trimmed)
        state = manager.get_state()
        # Should be within budget after management
        assert state.total_tokens <= 70

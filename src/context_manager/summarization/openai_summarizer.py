"""OpenAI-based conversation summarization strategy.

Uses OpenAI's chat completions API to generate concise summaries
of conversation segments that exceed the token budget.
"""

import logging
from collections.abc import Sequence

from openai import AsyncOpenAI

from context_manager.exceptions import ProviderAuthenticationError, SummarizationError
from context_manager.models import Message, Priority, Role

# Module-level logger for summarization diagnostics
logger = logging.getLogger(__name__)

# Default prompt template for conversation summarization
DEFAULT_PROMPT_TEMPLATE = (
    "Summarize the following conversation concisely, "
    "preserving key context and decisions:\n\n{messages}"
)


class OpenAISummarizer:
    """Summarizes conversation messages using OpenAI's chat API.

    Generates concise summaries of older messages to preserve
    context while reducing token usage in the context window.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_summary_tokens: int = 500,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        client: AsyncOpenAI | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the OpenAI summarizer.

        Args:
            model: OpenAI model ID to use for summarization.
            max_summary_tokens: Maximum tokens for the summary output.
            prompt_template: Template with {messages} placeholder.
            client: Optional pre-configured AsyncOpenAI client.
            timeout: Request timeout in seconds.
        """
        # Store the model identifier for API calls
        self._model = model
        # Store the maximum output token limit for summaries
        self._max_tokens = max_summary_tokens
        # Store the prompt template for formatting messages
        self._prompt_template = prompt_template
        # Use provided client or create from environment API key
        self._client = client or AsyncOpenAI()
        # Store the timeout for API request limits
        self._timeout = timeout
        # Log initialization configuration
        logger.debug(
            "OpenAISummarizer initialized (model=%s, max_tokens=%d)",
            model,
            max_summary_tokens,
        )

    async def summarize(self, messages: Sequence[Message]) -> Message:
        """Summarize a sequence of messages into a single condensed message.

        Calls the OpenAI chat completions API with a summarization prompt
        to generate a concise version of the conversation.

        Args:
            messages: The messages to summarize.

        Returns:
            A single Message containing the conversation summary.

        Raises:
            SummarizationError: If the API call fails.
        """
        # Validate that there are messages to summarize
        if not messages:
            # Raise an error for empty input since there's nothing to summarize
            raise SummarizationError("Cannot summarize an empty message list")
        # Format the messages into a readable text block
        formatted_text = self._format_messages(messages)
        # Build the summarization prompt using the template
        prompt = self._prompt_template.format(messages=formatted_text)
        # Log the summarization attempt with message count
        logger.info("Summarizing %d messages via OpenAI (%s)", len(messages), self._model)
        # Call the OpenAI API to generate the summary
        try:
            # Make the chat completions API call with the prompt
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
                temperature=0.3,
                timeout=self._timeout,
            )
            # Extract the summary text from the API response
            summary_content = response.choices[0].message.content
            # Validate that the response contains actual content
            if not summary_content:
                raise SummarizationError("OpenAI returned empty summary response")
        except SummarizationError:
            # Re-raise our own exceptions without wrapping
            raise
        except Exception as exc:
            # Check for authentication-specific errors
            if "authentication" in str(exc).lower() or "api key" in str(exc).lower():
                raise ProviderAuthenticationError(
                    f"OpenAI authentication failed: {exc}"
                ) from exc
            # Wrap all other errors in our summarization exception
            raise SummarizationError(f"OpenAI summarization failed: {exc}") from exc
        # Log successful summarization with character counts
        logger.debug(
            "Summary generated: %d chars from %d messages",
            len(summary_content),
            len(messages),
        )
        # Return the summary as a system message with high priority
        return Message(
            role=Role.SYSTEM,
            content=f"[Summary of previous conversation]\n{summary_content}",
            priority=Priority.HIGH,
            metadata={"source": "summarization", "original_count": str(len(messages))},
        )

    def _format_messages(self, messages: Sequence[Message]) -> str:
        """Format messages into a readable text block for the prompt.

        Args:
            messages: Messages to format.

        Returns:
            A formatted string with role labels and content.
        """
        # Build a list of formatted message strings
        lines: list[str] = []
        # Format each message with its role as a label
        for message in messages:
            # Capitalize the role for readability in the prompt
            role_label = message.role.value.capitalize()
            # Append the formatted line with role and content
            lines.append(f"{role_label}: {message.content}")
        # Join all formatted lines with newlines
        return "\n".join(lines)

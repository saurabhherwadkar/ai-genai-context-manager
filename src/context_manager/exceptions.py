"""Custom exception hierarchy for the context manager package.

All exceptions inherit from ContextManagerError to allow
callers to catch all package-specific errors with a single handler.
"""


class ContextManagerError(Exception):
    """Base exception for all context manager errors.

    All custom exceptions in this package inherit from this class,
    enabling callers to use a single except clause for package errors.
    """

    # Store the original error message for structured logging
    def __init__(self, message: str = "") -> None:
        # Call parent constructor with the error message
        self.message = message
        # Pass message up to the base Exception class
        super().__init__(message)


class ConfigurationError(ContextManagerError):
    """Raised when configuration is invalid, missing, or cannot be parsed.

    Examples: missing YAML file, invalid field values, schema violations.
    """


class TokenCountError(ContextManagerError):
    """Raised when token counting fails unexpectedly.

    Examples: encoding not found, API call to count endpoint failed.
    """


class TokenBudgetExceededError(ContextManagerError):
    """Raised when a single message exceeds the entire available token budget.

    This indicates the message cannot fit in any context window configuration.
    """

    # Store the offending token count for diagnostics
    def __init__(self, message: str = "", token_count: int = 0, budget: int = 0) -> None:
        # Save the actual token count that exceeded the budget
        self.token_count = token_count
        # Save the maximum allowed budget for error reporting
        self.budget = budget
        # Build a descriptive message if none provided
        if not message:
            message = f"Message requires {token_count} tokens but budget is {budget}"
        # Initialize the parent exception with the message
        super().__init__(message)


class TrimmingError(ContextManagerError):
    """Raised when the trimming strategy cannot reduce tokens sufficiently.

    This occurs when even after removing all trimmable messages,
    the remaining critical messages exceed the token budget.
    """


class SummarizationError(ContextManagerError):
    """Raised when summarization fails due to API error, timeout, or invalid response.

    Callers should fall back to pure trimming when this occurs.
    """


class ProviderError(ContextManagerError):
    """Raised when an LLM provider API call fails for any reason.

    This is the base class for all provider-specific errors.
    """


class ProviderAuthenticationError(ProviderError):
    """Raised when provider credentials are invalid or expired.

    The caller should verify their API key environment variables.
    """


class ProviderRateLimitError(ProviderError):
    """Raised when the provider's rate limit has been exceeded.

    Callers should implement backoff or retry logic.
    """

    # Store retry timing information from the provider response
    def __init__(self, message: str = "", retry_after_seconds: float | None = None) -> None:
        # Save the suggested retry delay from the provider headers
        self.retry_after_seconds = retry_after_seconds
        # Initialize parent with the error message
        super().__init__(message)


class ProviderTimeoutError(ProviderError):
    """Raised when a provider API call exceeds the configured timeout.

    Callers may retry with a longer timeout or smaller payload.
    """


class ValidationError(ContextManagerError):
    """Raised when input validation fails on user-provided data.

    Examples: message content too long, too many messages, invalid role.
    """

"""Unit tests for the custom exception hierarchy."""


from context_manager.exceptions import (
    ConfigurationError,
    ContextManagerError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SummarizationError,
    TokenBudgetExceededError,
    TokenCountError,
    TrimmingError,
    ValidationError,
)


class TestContextManagerError:
    """Tests for the base ContextManagerError exception."""

    def test_base_exception_with_message(self) -> None:
        """Verify the base exception stores and returns the message."""
        # Create an exception with a specific message
        exc = ContextManagerError("test error")
        # Assert the message is accessible via the message attribute
        assert exc.message == "test error"
        # Assert str() returns the message
        assert str(exc) == "test error"

    def test_base_exception_empty_message(self) -> None:
        """Verify the base exception handles empty messages."""
        # Create an exception with no message
        exc = ContextManagerError()
        # Assert the message defaults to empty string
        assert exc.message == ""

    def test_all_exceptions_inherit_from_base(self) -> None:
        """Verify all custom exceptions inherit from ContextManagerError."""
        # List all exception classes that should inherit from the base
        exception_classes = [
            ConfigurationError,
            TokenCountError,
            TokenBudgetExceededError,
            TrimmingError,
            SummarizationError,
            ProviderError,
            ProviderAuthenticationError,
            ProviderRateLimitError,
            ProviderTimeoutError,
            ValidationError,
        ]
        # Verify each class is a subclass of ContextManagerError
        for exc_class in exception_classes:
            assert issubclass(exc_class, ContextManagerError)


class TestTokenBudgetExceededError:
    """Tests for the TokenBudgetExceededError exception."""

    def test_stores_token_count_and_budget(self) -> None:
        """Verify the exception stores token count and budget values."""
        # Create an exception with specific token details
        exc = TokenBudgetExceededError(token_count=5000, budget=4096)
        # Assert both values are stored correctly
        assert exc.token_count == 5000
        assert exc.budget == 4096

    def test_auto_generates_message(self) -> None:
        """Verify a descriptive message is generated when none provided."""
        # Create without explicit message
        exc = TokenBudgetExceededError(token_count=5000, budget=4096)
        # Assert the auto-generated message contains the values
        assert "5000" in str(exc)
        assert "4096" in str(exc)

    def test_custom_message_overrides_auto(self) -> None:
        """Verify a custom message takes precedence over auto-generation."""
        # Create with an explicit custom message
        exc = TokenBudgetExceededError("custom msg", token_count=100, budget=50)
        # Assert the custom message is used
        assert str(exc) == "custom msg"


class TestProviderRateLimitError:
    """Tests for the ProviderRateLimitError exception."""

    def test_stores_retry_after(self) -> None:
        """Verify the exception stores the retry_after_seconds value."""
        # Create with retry timing information
        exc = ProviderRateLimitError("rate limited", retry_after_seconds=30.0)
        # Assert the retry value is stored
        assert exc.retry_after_seconds == 30.0

    def test_retry_after_defaults_to_none(self) -> None:
        """Verify retry_after_seconds defaults to None when not provided."""
        # Create without retry information
        exc = ProviderRateLimitError("rate limited")
        # Assert the default is None
        assert exc.retry_after_seconds is None

    def test_inherits_from_provider_error(self) -> None:
        """Verify ProviderRateLimitError inherits from ProviderError."""
        # Create the rate limit exception
        exc = ProviderRateLimitError("test")
        # Assert it can be caught as a ProviderError
        assert isinstance(exc, ProviderError)
        # Assert it can be caught as the base error
        assert isinstance(exc, ContextManagerError)


class TestExceptionCatchall:
    """Tests verifying the exception hierarchy allows catch-all handling."""

    def test_catch_all_provider_errors(self) -> None:
        """Verify all provider errors can be caught with ProviderError."""
        # Create each type of provider error
        errors = [
            ProviderError("generic"),
            ProviderAuthenticationError("auth"),
            ProviderRateLimitError("rate"),
            ProviderTimeoutError("timeout"),
        ]
        # Verify each is an instance of ProviderError
        for error in errors:
            assert isinstance(error, ProviderError)

    def test_catch_all_with_base_exception(self) -> None:
        """Verify all errors can be caught with ContextManagerError."""
        # Create one of each exception type
        errors = [
            ConfigurationError("config"),
            TokenCountError("count"),
            TrimmingError("trim"),
            SummarizationError("summarize"),
            ValidationError("validate"),
        ]
        # Verify each is an instance of the base exception
        for error in errors:
            assert isinstance(error, ContextManagerError)

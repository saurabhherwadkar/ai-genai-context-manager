"""Structured logging setup for the context manager package.

Configures Python's standard logging module based on the application's
YAML configuration. Supports both human-readable and structured JSON
output formats with configurable log levels.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from context_manager.config import LoggingConfig


class StructuredFormatter(logging.Formatter):
    """JSON lines formatter for structured log output.

    Produces one JSON object per log line, suitable for log aggregation
    systems like ELK, Datadog, or CloudWatch.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON object string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string representing the log entry.
        """
        # Build the base log entry dictionary with standard fields
        log_entry: dict[str, Any] = {
            # ISO 8601 timestamp in UTC for consistent time zone handling
            "timestamp": datetime.now(UTC).isoformat(),
            # Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            "level": record.levelname,
            # Logger name identifying the source module
            "logger": record.name,
            # The formatted log message content
            "message": record.getMessage(),
            # Source file where the log call originated
            "module": record.module,
            # Function name where the log call was made
            "function": record.funcName,
            # Line number in the source file
            "line": record.lineno,
        }
        # Include exception information if present on the record
        if record.exc_info and record.exc_info[0] is not None:
            # Format the full traceback as a string for the JSON output
            log_entry["exception"] = self.formatException(record.exc_info)
        # Serialize the dictionary to a compact JSON string
        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for development and local debugging.

    Produces colored, easy-to-scan log lines with timestamp and context.
    """

    # Format string template with timestamp, level, logger, and message
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    # Date format for the timestamp portion of log lines
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        """Initialize the formatter with the predefined format string."""
        # Pass the format template and date format to the parent class
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


def setup_logging(config: LoggingConfig) -> None:
    """Configure the logging system based on the provided configuration.

    Sets up handlers, formatters, and log levels for the context_manager
    package logger hierarchy.

    Args:
        config: LoggingConfig instance with level, format, and file settings.
    """
    # Get the root logger for the context_manager package
    package_logger = logging.getLogger("context_manager")
    # Set the log level from the configuration string
    package_logger.setLevel(config.level)
    # Remove any existing handlers to prevent duplicate output
    package_logger.handlers.clear()
    # Select the appropriate formatter based on configuration
    if config.format == "structured":
        # Use JSON lines format for production log aggregation
        formatter: logging.Formatter = StructuredFormatter()
    else:
        # Use human-readable format for development environments
        formatter = HumanFormatter()
    # Always add a stderr handler for console output
    console_handler = logging.StreamHandler(sys.stderr)
    # Apply the selected formatter to the console handler
    console_handler.setFormatter(formatter)
    # Register the console handler with the package logger
    package_logger.addHandler(console_handler)
    # Optionally add a file handler if a log file path is configured
    if config.file:
        # Create a file handler that appends to the specified path
        file_handler = logging.FileHandler(config.file, encoding="utf-8")
        # Apply the same formatter for consistency across outputs
        file_handler.setFormatter(formatter)
        # Register the file handler with the package logger
        package_logger.addHandler(file_handler)
    # Prevent log messages from propagating to the root logger
    package_logger.propagate = False

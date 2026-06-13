"""Message trimming strategies for context window management.

This subpackage provides multiple trimming implementations:
- FifoTrimmingStrategy: Removes oldest messages first
- SlidingWindowStrategy: Keeps only the N most recent messages
- PriorityTrimmingStrategy: Removes lowest-priority messages first
"""

from context_manager.trimming.fifo import FifoTrimmingStrategy
from context_manager.trimming.priority import PriorityTrimmingStrategy
from context_manager.trimming.sliding_window import SlidingWindowStrategy

# Export all trimming strategy implementations
__all__ = [
    "FifoTrimmingStrategy",
    "PriorityTrimmingStrategy",
    "SlidingWindowStrategy",
]

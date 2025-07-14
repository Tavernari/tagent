"""
TAgent UI Module - Unified interface for different UI styles.

This module provides a unified interface for TAgent UI implementations,
supporting both animated (Matrix-style) and institutional (server-focused) UI styles.
"""

from typing import Optional

from .factory import UIFactory, UIStyle
from .interface import UIInterface, MessageType
from .animated import Colors


# Global UI instance
_ui_instance: Optional[UIInterface] = None


def get_ui() -> UIInterface:
    """Get the current UI instance, creating one if none exists."""
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = UIFactory.create_ui()
    return _ui_instance


def set_ui_style(style: UIStyle) -> None:
    """Set the UI style and recreate the UI instance."""
    global _ui_instance
    _ui_instance = UIFactory.create_ui(style)


def start_thinking(message: str = "Thinking") -> None:
    """Start thinking animation or indicator."""
    get_ui().start_thinking(message)


def stop_thinking() -> None:
    """Stop thinking animation or indicator."""
    get_ui().stop_thinking()


def print_retro_banner(text: str, char: str = "=", width: int = 60, message_type: MessageType = MessageType.PRIMARY) -> None:
    """Print a banner with the given text."""
    get_ui().print_banner(text, char, width, message_type)


def print_retro_step(step_num: int, action: str, title: str) -> None:
    """Print a step with action and title."""
    get_ui().print_step(step_num, action, title)


def print_retro_status(status: str, details: str = "", message_type: Optional[MessageType] = None) -> None:
    """Print status messages with optional semantic type."""
    get_ui().print_status(status, details, message_type)


def print_plan_details(content: str, max_width: int = 80) -> None:
    """Print plan details."""
    get_ui().print_plan_details(content, max_width)


def print_feedback_dimmed(feedback_type: str, content: str, max_length: Optional[int] = None) -> None:
    """Print feedback in dimmed text."""
    get_ui().print_feedback_dimmed(feedback_type, content, max_length)


def print_retro_progress_bar(current: int, total: int, width: int = 30) -> None:
    """Print a progress bar."""
    get_ui().print_progress_bar(current, total, width)


# Legacy compatibility functions (keeping the same names)
print_banner = print_retro_banner
print_step = print_retro_step
print_status = print_retro_status
print_progress_bar = print_retro_progress_bar

# Export important classes for direct use
__all__ = [
    'UIInterface',
    'MessageType',
    'UIStyle',
    'UIFactory',
    'Colors',
    'get_ui',
    'set_ui_style',
    'start_thinking',
    'stop_thinking',
    'print_retro_banner',
    'print_retro_step',
    'print_retro_status',
    'print_plan_details',
    'print_feedback_dimmed',
    'print_retro_progress_bar',
    'print_banner',
    'print_step',
    'print_status',
    'print_progress_bar',
]
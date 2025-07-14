from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum


class MessageType(Enum):
    """Semantic message types for UI styling."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    THINKING = "thinking"
    EXECUTE = "execute"
    PLAN = "plan"
    EVALUATE = "evaluate"
    SUMMARIZE = "summarize"
    FORMAT = "format"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    MUTED = "muted"


class UIInterface(ABC):
    """Base interface for TAgent UI implementations."""
    
    @abstractmethod
    def start_thinking(self, message: str = "Thinking") -> None:
        """Start thinking animation or indicator."""
        pass
    
    @abstractmethod
    def stop_thinking(self) -> None:
        """Stop thinking animation or indicator."""
        pass
    
    @abstractmethod
    def print_banner(self, text: str, char: str = "=", width: int = 60, message_type: MessageType = MessageType.PRIMARY) -> None:
        """Print a banner with the given text."""
        pass
    
    @abstractmethod
    def print_step(self, step_num: int, action: str, title: str) -> None:
        """Print a step with action and title."""
        pass
    
    @abstractmethod
    def print_status(self, status: str, details: str = "", message_type: Optional[MessageType] = None) -> None:
        """Print status messages with optional semantic type."""
        pass
    
    @abstractmethod
    def print_plan_details(self, content: str, max_width: int = 80) -> None:
        """Print plan details."""
        pass
    
    @abstractmethod
    def print_feedback_dimmed(self, feedback_type: str, content: str, max_length: Optional[int] = None) -> None:
        """Print feedback in dimmed text."""
        pass
    
    @abstractmethod
    def print_progress_bar(self, current: int, total: int, width: int = 30) -> None:
        """Print a progress bar."""
        pass
import time
import textwrap
import shutil
from typing import Optional

from .interface import UIInterface, MessageType


class InstitutionalUI(UIInterface):
    """Server-focused institutional UI implementation with structured logging."""
    
    def __init__(self):
        self._thinking_active = False
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp for logging."""
        return time.strftime('%Y-%m-%d %H:%M:%S')
    
    def _get_log_level_for_message_type(self, message_type: MessageType) -> str:
        """Map semantic message types to log levels."""
        level_map = {
            MessageType.SUCCESS: "INFO",
            MessageType.ERROR: "ERROR",
            MessageType.WARNING: "WARN",
            MessageType.INFO: "INFO",
            MessageType.THINKING: "DEBUG",
            MessageType.EXECUTE: "INFO",
            MessageType.PLAN: "INFO",
            MessageType.EVALUATE: "INFO",
            MessageType.SUMMARIZE: "INFO",
            MessageType.FORMAT: "DEBUG",
            MessageType.PRIMARY: "INFO",
            MessageType.SECONDARY: "DEBUG",
            MessageType.MUTED: "DEBUG",
        }
        return level_map.get(message_type, "INFO")
    
    def _log_message(self, level: str, component: str, message: str) -> None:
        """Log a structured message."""
        timestamp = self._get_timestamp()
        print(f"{timestamp} [{level:5}] {component}: {message}")
    
    def start_thinking(self, message: str = "Thinking") -> None:
        """Start thinking indicator."""
        self._thinking_active = True
        self._log_message("DEBUG", "AGENT", f"Processing: {message}")
    
    def stop_thinking(self) -> None:
        """Stop thinking indicator."""
        if self._thinking_active:
            self._thinking_active = False
            self._log_message("DEBUG", "AGENT", "Processing completed")
    
    def print_banner(self, text: str, char: str = "=", width: int = 60, message_type: MessageType = MessageType.PRIMARY) -> None:
        """Print a structured banner."""
        level = self._get_log_level_for_message_type(message_type)
        border = char * width
        padding = (width - len(text) - 2) // 2
        padded_text = " " * padding + text + " " * padding
        if len(padded_text) < width - 2:
            padded_text += " "
        
        print(border)
        print(f"{char}{padded_text}{char}")
        print(border)
        self._log_message(level, "SYSTEM", f"Banner: {text}")
    
    def print_step(self, step_num: int, action: str, title: str) -> None:
        """Print a structured step."""
        step_text = f"STEP {step_num:02d}: {action.upper()}"
        self._log_message("INFO", "WORKFLOW", step_text)
        if title:
            self._log_message("INFO", "WORKFLOW", f"Task: {title}")
    
    def print_status(self, status: str, details: str = "", message_type: Optional[MessageType] = None) -> None:
        """Print structured status messages."""
        status_upper = status.upper()
        
        if message_type is None:
            # Auto-detect message type from status
            status_message_types = {
                "SUCCESS": MessageType.SUCCESS,
                "ERROR": MessageType.ERROR,
                "WARNING": MessageType.WARNING,
                "THINKING": MessageType.THINKING,
                "EXECUTE": MessageType.EXECUTE,
                "PLAN": MessageType.PLAN,
                "EVALUATE": MessageType.EVALUATE,
                "SUMMARIZE": MessageType.SUMMARIZE,
                "FORMAT": MessageType.FORMAT,
            }
            message_type = status_message_types.get(status_upper, MessageType.INFO)
        
        level = self._get_log_level_for_message_type(message_type)
        component = status_upper if status_upper in ["EXECUTE", "PLAN", "EVALUATE", "SUMMARIZE", "FORMAT"] else "AGENT"
        
        if details:
            self._log_message(level, component, f"{status_upper}: {details}")
        else:
            self._log_message(level, component, status_upper)
    
    def print_plan_details(self, content: str, max_width: int = 80) -> None:
        """Print structured plan details."""
        if not content:
            return
        
        clean_content = " ".join(content.split())
        self._log_message("INFO", "PLANNER", f"Plan: {clean_content}")
        
        # For long plans, log additional details
        if len(clean_content) > max_width:
            words = clean_content.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_width:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            for i, chunk in enumerate(chunks[1:], 1):
                self._log_message("DEBUG", "PLANNER", f"Plan (cont {i}): {chunk}")
    
    def print_feedback_dimmed(self, feedback_type: str, content: str, max_length: Optional[int] = None) -> None:
        """Print structured feedback."""
        if not content:
            return
        
        terminal_width = shutil.get_terminal_size().columns
        if max_length is None:
            max_length = terminal_width - 40  # Account for log prefix
        
        wrapped_content = textwrap.wrap(content, width=max_length)
        
        for line in wrapped_content:
            if feedback_type == "FEEDBACK":
                self._log_message("DEBUG", "FEEDBACK", line)
            elif feedback_type == "MISSING":
                self._log_message("WARN", "VALIDATION", f"Missing: {line}")
            elif feedback_type == "SUGGESTIONS":
                self._log_message("INFO", "SUGGESTIONS", line)
            else:
                self._log_message("DEBUG", "FEEDBACK", line)
    
    def print_progress_bar(self, current: int, total: int, width: int = 30) -> None:
        """Print structured progress information."""
        percentage = int(100 * current / total)
        self._log_message("INFO", "PROGRESS", f"Progress: {percentage}% ({current}/{total})")
        
        # Also show a simple progress bar
        filled = int(width * current / total)
        bar = f"[{'=' * filled}{'-' * (width - filled)}]"
        print(f"{bar} {percentage:3d}% ({current}/{total})")
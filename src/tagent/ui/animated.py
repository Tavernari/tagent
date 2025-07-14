import threading
import time
import sys
import textwrap
import shutil
from typing import Optional

from .interface import UIInterface, MessageType


TERMINAL_MODE = "dark"


class Colors:
    """ANSI color codes based on terminal mode."""
    if TERMINAL_MODE == "dark":
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        BRIGHT_BLACK = "\033[90m"  # Light gray
        BRIGHT_RED = "\033[91m"
        BRIGHT_GREEN = "\033[92m"
        BRIGHT_YELLOW = "\033[93m"
        BRIGHT_BLUE = "\033[94m"
        BRIGHT_MAGENTA = "\033[95m"
        BRIGHT_CYAN = "\033[96m"
        BRIGHT_WHITE = "\033[97m"
    else:  # Light mode
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        BRIGHT_BLACK = "\033[90m"
        BRIGHT_RED = "\033[91m"
        BRIGHT_GREEN = "\033[92m"
        BRIGHT_YELLOW = "\033[93m"
        BRIGHT_BLUE = "\033[94m"
        BRIGHT_MAGENTA = "\033[95m"
        BRIGHT_CYAN = "\033[96m"
        BRIGHT_WHITE = "\033[97m"


class ThinkingAnimation:
    """Thinking animation thread that runs until stopped."""
    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=0.5)
            sys.stdout.write(f"\r{' ' * (len(self.message) + 10)}\r")
            sys.stdout.flush()

    def _animate(self):
        i = 0
        while self.running:
            dots = "." * ((i % 4) + 1)
            sys.stdout.write(f"\r{Colors.GREEN}[*] {self.message}{dots:<4}{Colors.RESET}")
            sys.stdout.flush()
            time.sleep(0.25)
            i += 1


class AnimatedUI(UIInterface):
    """Matrix-style animated UI implementation."""
    
    def __init__(self):
        self._thinking_animation = None
    
    def _get_color_for_message_type(self, message_type: MessageType) -> str:
        """Map semantic message types to ANSI colors."""
        color_map = {
            MessageType.SUCCESS: Colors.BRIGHT_GREEN,
            MessageType.ERROR: Colors.BRIGHT_RED,
            MessageType.WARNING: Colors.BRIGHT_YELLOW,
            MessageType.INFO: Colors.BRIGHT_CYAN,
            MessageType.THINKING: Colors.GREEN,
            MessageType.EXECUTE: Colors.BRIGHT_GREEN,
            MessageType.PLAN: Colors.BRIGHT_GREEN,
            MessageType.EVALUATE: Colors.BRIGHT_GREEN,
            MessageType.SUMMARIZE: Colors.BRIGHT_GREEN,
            MessageType.FORMAT: Colors.BRIGHT_GREEN,
            MessageType.PRIMARY: Colors.BRIGHT_GREEN,
            MessageType.SECONDARY: Colors.GREEN,
            MessageType.MUTED: Colors.DIM + Colors.GREEN,
        }
        return color_map.get(message_type, Colors.GREEN)
    
    def start_thinking(self, message: str = "Thinking") -> None:
        """Start thinking animation."""
        self.stop_thinking()
        self._thinking_animation = ThinkingAnimation(message)
        self._thinking_animation.start()
    
    def stop_thinking(self) -> None:
        """Stop thinking animation."""
        if self._thinking_animation:
            self._thinking_animation.stop()
            self._thinking_animation = None
    
    def _type_line(self, line: str, color: str, typing_speed: float = 0.002, blink_duration: float = 0.05, blink_speed: float = 0.01) -> None:
        """Writes a line quickly with blinking cursor at the end."""
        sys.stdout.write(color)
        for char in line:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(typing_speed)
        # Simple cursor blink - reduced for performance
        num_blinks = int(blink_duration / (2 * blink_speed))
        for _ in range(num_blinks):
            sys.stdout.write('|')
            sys.stdout.flush()
            time.sleep(blink_speed)
            sys.stdout.write('\b \b')
            sys.stdout.flush()
            time.sleep(blink_speed)
        sys.stdout.write(Colors.RESET + '\n')
    
    def print_banner(self, text: str, char: str = "=", width: int = 60, message_type: MessageType = MessageType.PRIMARY) -> None:
        """Prints a retro-style banner with typing effect."""
        color = self._get_color_for_message_type(message_type)
        border = char * width
        padding = (width - len(text) - 2) // 2
        padded_text = " " * padding + text + " " * padding
        if len(padded_text) < width - 2:
            padded_text += " "
        self._type_line(border, color)
        self._type_line(f"{char}{padded_text}{char}", color)
        self._type_line(border, color)
    
    def print_step(self, step_num: int, action: str, title: str) -> None:
        """Prints a step with retro-style typing effect."""
        action_message_types = {
            "EXECUTE": MessageType.EXECUTE,
            "PLAN": MessageType.PLAN,
            "SUMMARIZE": MessageType.SUMMARIZE,
            "EVALUATE": MessageType.EVALUATE,
        }
        message_type = action_message_types.get(action.upper(), MessageType.PRIMARY)
        color = self._get_color_for_message_type(message_type)
        step_text = f"STEP {step_num:02d}: {action.upper()}"
        self._type_line(step_text, color)
        self._type_line(title, self._get_color_for_message_type(MessageType.MUTED))
    
    def print_status(self, status: str, details: str = "", message_type: Optional[MessageType] = None) -> None:
        """Prints status messages with retro-style typing effect."""
        timestamp = f"[{__import__('time').strftime('%H:%M:%S')}]"
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
        
        status_symbols = {
            MessageType.SUCCESS: "[+]",
            MessageType.ERROR: "[!]",
            MessageType.WARNING: "[~]",
            MessageType.THINKING: "[*]",
            MessageType.EXECUTE: "[>]",
            MessageType.PLAN: "[#]",
            MessageType.EVALUATE: "[?]",
            MessageType.SUMMARIZE: "[=]",
            MessageType.FORMAT: "[@]",
        }
        
        symbol = status_symbols.get(message_type, "[-]")
        color = self._get_color_for_message_type(message_type)
        message = f"{symbol} {timestamp} {status_upper}: {details}"
        self._type_line(message, color)
    
    def print_plan_details(self, content: str, max_width: int = 80) -> None:
        """Prints plan details with typing effect."""
        if not content:
            return
        clean_content = " ".join(content.split())
        color = self._get_color_for_message_type(MessageType.MUTED)
        
        if len(clean_content) <= max_width:
            self._type_line(f"  * Plan: {clean_content}", color)
            return
        
        first_line = clean_content[:max_width-10] + "..."
        self._type_line(f"  * Plan: {first_line}", color)
        remaining = clean_content[max_width-10:]
        words = remaining.split()
        current_line = "        "
        
        for word in words[:15]:
            if len(current_line + word + " ") <= max_width:
                current_line += word + " "
            else:
                if current_line.strip():
                    self._type_line(current_line.rstrip(), color)
                current_line = "        " + word + " "
        
        if current_line.strip() and len(current_line) > 8:
            if len(words) > 15:
                current_line = current_line.rstrip() + "..."
            self._type_line(current_line.rstrip(), color)
    
    def print_feedback_dimmed(self, feedback_type: str, content: str, max_length: Optional[int] = None) -> None:
        """Prints feedback in dimmed text, wrapping long lines."""
        if not content:
            return
        
        terminal_width = shutil.get_terminal_size().columns
        if max_length is None:
            max_length = terminal_width - 10
        
        wrapped_content = textwrap.wrap(content, width=max_length)
        color = self._get_color_for_message_type(MessageType.MUTED)
        
        for line in wrapped_content:
            if feedback_type == "FEEDBACK":
                print(f"{color}   - {line}{Colors.RESET}")
            elif feedback_type == "MISSING":
                print(f"{color}   # Missing: {line}{Colors.RESET}")
            elif feedback_type == "SUGGESTIONS":
                print(f"{color}   * {line}{Colors.RESET}")
            else:
                print(f"{color}   I {line}{Colors.RESET}")
    
    def print_progress_bar(self, current: int, total: int, width: int = 30) -> None:
        """Prints a simple progress bar without special characters."""
        filled = int(width * current / total)
        success_color = self._get_color_for_message_type(MessageType.SUCCESS)
        secondary_color = self._get_color_for_message_type(MessageType.SECONDARY)
        
        bar = f"{success_color}{'=' * filled}{secondary_color}{'-' * (width - filled)}{Colors.RESET}"
        percentage = int(100 * current / total)
        print(f"[{bar}] {success_color}{percentage:3d}%{Colors.RESET} ({current}/{total})")
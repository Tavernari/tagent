import threading
import time
import sys
import textwrap
import shutil
from typing import Optional

from .interface import UIInterface, MessageType


class Colors:
    """ANSI color codes for a modern, VSCode-like terminal experience."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # VSCode Dark+ inspired palette
    BLACK = "\033[30m"
    RED = "\033[38;2;241;76;76m"       # Editor error foreground
    GREEN = "\033[38;2;110;219;123m"    # Git decoration added
    YELLOW = "\033[38;2;223;175;95m"   # Editor warning foreground
    BLUE = "\033[38;2;79;193;255m"      # Editor info foreground
    MAGENTA = "\033[38;2;198;120;221m"  # Keyword color
    CYAN = "\033[38;2;86;182;194m"       # String color
    WHITE = "\033[37m"
    
    # Bright variants
    BRIGHT_BLACK = "\033[90m"  # Comments, muted text
    BRIGHT_RED = "\033[38;2;250;100;100m"
    BRIGHT_GREEN = "\033[38;2;130;239;143m"
    BRIGHT_YELLOW = "\033[38;2;243;195;115m"
    BRIGHT_BLUE = "\033[38;2;99;213;255m"
    BRIGHT_MAGENTA = "\033[38;2;218;140;241m"
    BRIGHT_CYAN = "\033[38;2;106;202;214m"
    BRIGHT_WHITE = "\033[97m"

    # Semantic aliases
    PRIMARY = BLUE
    SECONDARY = CYAN
    SUCCESS = GREEN
    WARNING = YELLOW
    ERROR = RED
    INFO = BRIGHT_CYAN
    MUTED = BRIGHT_BLACK


class TAgentThinkingAnimation:
    """A branded, professional thinking animation for TAgent."""
    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.running = False
        self.thread = None
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.animation_line = ""
        self.bar_width = 12
        self.shuttle_pos = 0
        self.shuttle_dir = 1

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
            # Clear the animation line
            sys.stdout.write('\r' + ' ' * (len(self.animation_line) + 5) + '\r')
            sys.stdout.flush()

    def _animate(self):
        spinner_idx = 0
        while self.running:
            spinner_char = self.spinner_chars[spinner_idx % len(self.spinner_chars)]
            timestamp = time.strftime('%H:%M:%S')

            # Update shuttle position
            if self.shuttle_pos >= self.bar_width - 1:
                self.shuttle_dir = -1
            elif self.shuttle_pos <= 0:
                self.shuttle_dir = 1
            self.shuttle_pos += self.shuttle_dir

            # Build the bar
            bar = ['·'] * self.bar_width
            shuttle_char = '»' if self.shuttle_dir == 1 else '«'
            bar[self.shuttle_pos] = f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{shuttle_char}{Colors.RESET}{Colors.CYAN}"
            bar_str = "".join(bar)

            # Assemble the final line
            self.animation_line = (
                f"{Colors.BOLD}{Colors.PRIMARY}{spinner_char}{Colors.RESET} "
                f"{Colors.MUTED}[{timestamp}]{Colors.RESET} "
                f"{Colors.ITALIC}{self.message}{Colors.RESET} "
                f"[{Colors.BOLD}T{Colors.RESET}"
                f"{Colors.CYAN}{bar_str}{Colors.RESET}]"
            )

            sys.stdout.write('\r' + self.animation_line)
            sys.stdout.flush()

            time.sleep(0.08)
            spinner_idx += 1


class ModernCliUI(UIInterface):
    """A modern, professional CLI UI inspired by VSCode."""
    
    def __init__(self):
        self._thinking_animation = None
        self._terminal_width = shutil.get_terminal_size().columns
    
    def _get_color_for_message_type(self, message_type: MessageType) -> str:
        """Map semantic message types to the modern color palette."""
        color_map = {
            MessageType.SUCCESS: Colors.SUCCESS,
            MessageType.ERROR: Colors.ERROR,
            MessageType.WARNING: Colors.WARNING,
            MessageType.INFO: Colors.INFO,
            MessageType.THINKING: Colors.PRIMARY,
            MessageType.EXECUTE: Colors.PRIMARY,
            MessageType.PLAN: Colors.MAGENTA,
            MessageType.EVALUATE: Colors.CYAN,
            MessageType.SUMMARIZE: Colors.GREEN,
            MessageType.FORMAT: Colors.BLUE,
            MessageType.PRIMARY: Colors.PRIMARY,
            MessageType.SECONDARY: Colors.SECONDARY,
            MessageType.MUTED: Colors.MUTED,
        }
        return color_map.get(message_type, Colors.PRIMARY)

    def start_thinking(self, message: str = "Thinking") -> None:
        self.stop_thinking()
        self._thinking_animation = TAgentThinkingAnimation(message)
        self._thinking_animation.start()
    
    def stop_thinking(self) -> None:
        if self._thinking_animation:
            self._thinking_animation.stop()
            self._thinking_animation = None

    def print_banner(self, text: str, char: str = "─", width: int = 60, message_type: MessageType = MessageType.PRIMARY) -> None:
        color = self._get_color_for_message_type(message_type)
        
        top_border = f"╭{char * (width - 2)}╮"
        bottom_border = f"╰{char * (width - 2)}╯"
        
        padding = (width - len(text) - 2) // 2
        padded_text = " " * padding + text + " " * (width - 2 - len(text) - padding)
        main_line = f"│{padded_text}│"

        # Animate the banner drawing
        for line in [top_border, main_line, bottom_border]:
            print(f"{color}{Colors.BOLD}{line}{Colors.RESET}")
            time.sleep(0.05)
        
        print() # Add a blank line for spacing

    def print_step(self, step_num: int, action: str, title: str) -> None:
        action_color = self._get_color_for_message_type(MessageType.PLAN)
        step_prefix = f"Step {step_num:02d}"
        
        print(
            f"{Colors.BOLD}{action_color}{step_prefix}:{Colors.RESET} "
            f"{Colors.BOLD}{action.upper()}{Colors.RESET}"
        )
        print(f"{Colors.MUTED}  └─ {title}{Colors.RESET}")

    def print_status(self, status: str, details: str = "", message_type: Optional[MessageType] = None) -> None:
        timestamp = time.strftime('%H:%M:%S')
        status_upper = status.upper()
        
        if message_type is None:
            status_map = {
                "SUCCESS": MessageType.SUCCESS, "ERROR": MessageType.ERROR,
                "WARNING": MessageType.WARNING, "INFO": MessageType.INFO,
                "EXECUTE": MessageType.EXECUTE, "PLAN": MessageType.PLAN,
                "EVALUATE": MessageType.EVALUATE, "SUMMARY": MessageType.SUMMARIZE,
            }
            message_type = status_map.get(status_upper, MessageType.INFO)

        color = self._get_color_for_message_type(message_type)
        
        symbol_map = {
            MessageType.SUCCESS: "✔", MessageType.ERROR: "✖",
            MessageType.WARNING: "⚠", MessageType.INFO: "ℹ",
            MessageType.EXECUTE: "▶", MessageType.PLAN: "§",
            MessageType.EVALUATE: "?", MessageType.SUMMARIZE: "Σ",
        }
        symbol = symbol_map.get(message_type, "●")

        status_line = (
            f"{color}{Colors.BOLD}{symbol}{Colors.RESET} "
            f"{Colors.MUTED}[{timestamp}]{Colors.RESET} "
            f"{color}{status_upper}{Colors.RESET}"
        )
        
        if details:
            status_line += f": {details}"
            
        print(status_line)

    def print_plan_details(self, content: str, max_width: int = 80) -> None:
        if not content:
            return
        
        clean_content = " ".join(content.split())
        wrapped_lines = textwrap.wrap(clean_content, width=max_width - 4) # 4 for '  └─ '
        
        if wrapped_lines:
            print(f"{Colors.MUTED}  └─ {wrapped_lines[0]}{Colors.RESET}")
            for line in wrapped_lines[1:]:
                print(f"{Colors.MUTED}     {line}{Colors.RESET}")

    def print_feedback_dimmed(self, feedback_type: str, content: str, max_length: Optional[int] = None) -> None:
        if not content:
            return
        
        terminal_width = self._terminal_width
        if max_length is None:
            max_length = terminal_width - 10
        
        wrapped_content = textwrap.wrap(content, width=max_length)
        
        prefix_map = {
            "FEEDBACK": "  - ", "MISSING": "  # ", "SUGGESTIONS": "  * ",
            "TOKEN_USAGE": "", "MODEL_STATS": "", "TOTAL_STATS": ""
        }
        prefix = prefix_map.get(feedback_type, "  I ")
        
        for line in wrapped_content:
            print(f"{Colors.MUTED}{prefix}{line}{Colors.RESET}")

    def print_progress_bar(self, current: int, total: int, width: int = 30) -> None:
        filled = int(width * current / total)
        percentage = int(100 * current / total)
        
        bar = f"{Colors.SUCCESS}{'█' * filled}{Colors.MUTED}{'─' * (width - filled)}{Colors.RESET}"
        
        progress_line = f"  [{bar}] {percentage:3d}% ({current}/{total})"
        
        # Use carriage return to show progress on a single line
        sys.stdout.write('\r' + progress_line)
        if current == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

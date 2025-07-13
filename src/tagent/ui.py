"""
Terminal UI and logging functions for TAgent with 90s style aesthetics.
"""

import threading
import time
import sys


class Colors:
    """ANSI Color Codes for 90s terminal aesthetics."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Basic colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class ThinkingAnimation:
    """Threaded thinking animation that runs until stopped."""

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.running = False
        self.thread = None

    def start(self):
        """Start the thinking animation."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the thinking animation and clear the line."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=0.5)
            # Clear the thinking line
            sys.stdout.write(f"\r{' ' * (len(self.message) + 10)}\r")
            sys.stdout.flush()

    def _animate(self):
        """Internal animation loop."""
        i = 0
        while self.running:
            dots = "." * ((i % 4) + 1)
            sys.stdout.write(
                f"\r{Colors.CYAN}[*] {self.message}{dots:<4}{Colors.RESET}"
            )
            sys.stdout.flush()
            time.sleep(0.25)
            i += 1


# Global thinking animation instance
_thinking_animation = None


def start_thinking(message: str = "Thinking") -> None:
    """Start a persistent thinking animation."""
    global _thinking_animation
    stop_thinking()  # Stop any existing animation
    _thinking_animation = ThinkingAnimation(message)
    _thinking_animation.start()


def stop_thinking() -> None:
    """Stop the current thinking animation."""
    global _thinking_animation
    if _thinking_animation:
        _thinking_animation.stop()
        _thinking_animation = None


def print_retro_banner(
    text: str, char: str = "=", width: int = 60, color: str = Colors.BRIGHT_CYAN
) -> None:
    """Prints a retro-style banner with ASCII art and colors."""
    border = char * width
    padding = (width - len(text) - 2) // 2
    padded_text = " " * padding + text + " " * padding
    if len(padded_text) < width - 2:
        padded_text += " "

    print(f"\n{color}{border}")
    print(f"{char}{padded_text}{char}")
    print(f"{border}{Colors.RESET}")


def print_retro_step(step_num: int, action: str, title: str) -> None:
    """Prints a minimalist step indicator with dynamic ASCII art."""
    action_colors = {
        "EXECUTE": Colors.BRIGHT_GREEN,
        "PLAN": Colors.BRIGHT_YELLOW,
        "SUMMARIZE": Colors.BRIGHT_BLUE,
        "EVALUATE": Colors.BRIGHT_MAGENTA,
    }
    action_color = action_colors.get(action.upper(), Colors.WHITE)

    # Calculate dynamic width based on content
    step_text = f"STEP {step_num:02d}: {action.upper()}"
    max_width = max(len(step_text), len(title)) + 2  # Add padding
    border_width = max_width + 4

    # Dynamic ASCII art step indicator
    top_border = f"+{'-' * (border_width - 2)}+"
    bottom_border = f"+{'-' * (border_width - 2)}+"

    print(f"\n{Colors.BRIGHT_WHITE}{top_border}{Colors.RESET}")
    print(
        f"{Colors.BRIGHT_WHITE}| {action_color}{step_text:<{max_width}}{Colors.BRIGHT_WHITE} |{Colors.RESET}"
    )
    print(
        f"{Colors.BRIGHT_WHITE}| {Colors.DIM}{title:<{max_width}}{Colors.RESET}{Colors.BRIGHT_WHITE} |{Colors.RESET}"
    )
    print(f"{Colors.BRIGHT_WHITE}{bottom_border}{Colors.RESET}")


def print_retro_status(status: str, details: str = "") -> None:
    """Prints retro-style status messages with colors and ASCII art."""
    timestamp = f"[{__import__('time').strftime('%H:%M:%S')}]"

    if status == "SUCCESS":
        print(
            f"\n{Colors.BRIGHT_GREEN}[+] {timestamp} {status}: {details}{Colors.RESET}"
        )
    elif status == "ERROR":
        print(f"\n{Colors.BRIGHT_RED}[!] {timestamp} {status}: {details}{Colors.RESET}")
    elif status == "WARNING":
        print(
            f"\n{Colors.BRIGHT_YELLOW}[~] {timestamp} {status}: {details}{Colors.RESET}"
        )
    elif status == "THINKING":
        print(f"\n{Colors.CYAN}[*] {timestamp} {status}: {details}{Colors.RESET}")
    elif status == "EXECUTE":
        print(
            f"\n{Colors.BRIGHT_GREEN}[>] {timestamp} {status}: {details}{Colors.RESET}"
        )
    elif status == "PLAN":
        print(
            f"\n{Colors.BRIGHT_YELLOW}[#] {timestamp} {status}: {details}{Colors.RESET}"
        )
    elif status == "EVALUATE":
        print(
            f"\n{Colors.BRIGHT_MAGENTA}[?] {timestamp} {status}: {details}{Colors.RESET}"
        )
    elif status == "SUMMARIZE":
        print(
            f"\n{Colors.BRIGHT_BLUE}[=] {timestamp} {status}: {details}{Colors.RESET}"
        )
    elif status == "FORMAT":
        print(
            f"\n{Colors.BRIGHT_CYAN}[@] {timestamp} {status}: {details}{Colors.RESET}"
        )
    else:
        print(f"\n{Colors.WHITE}[-] {timestamp} {status}: {details}{Colors.RESET}")


def print_plan_details(content: str, max_width: int = 80) -> None:
    """Prints plan details with proper formatting and wrapping."""
    if not content:
        return
    
    # Clean up the content
    clean_content = " ".join(content.split())
    
    # If content is short enough, show in one line
    if len(clean_content) <= max_width:
        print(f"{Colors.DIM}   📝 Plan: {clean_content}{Colors.RESET}")
        return
    
    # For longer content, show first line and continuation
    first_line = clean_content[:max_width-10] + "..."
    print(f"{Colors.DIM}   📝 Plan: {first_line}{Colors.RESET}")
    
    # Show additional lines for better readability
    remaining = clean_content[max_width-10:]
    words = remaining.split()
    current_line = "        "  # Indent continuation
    
    for word in words[:15]:  # Show up to 15 more words
        if len(current_line + word + " ") <= max_width:
            current_line += word + " "
        else:
            if current_line.strip():
                print(f"{Colors.DIM}{current_line.rstrip()}{Colors.RESET}")
            current_line = "        " + word + " "
    
    if current_line.strip() and len(current_line) > 8:
        # Add final truncation if there's still more content
        if len(words) > 15:
            current_line = current_line.rstrip() + "..."
        print(f"{Colors.DIM}{current_line.rstrip()}{Colors.RESET}")


def print_feedback_dimmed(
    feedback_type: str, content: str, max_length: int = None
) -> None:
    """Prints feedback in a dimmed, non-verbose style for quick overview."""
    if not content:
        return

    # Special handling for plan feedback
    if feedback_type == "PLAN_FEEDBACK":
        print_plan_details(content)
        return

    # Set default max_length based on feedback type
    if max_length is None:
        max_length = 80   # Default for other feedback types

    # Truncate content if too long
    truncated = content[:max_length] + "..." if len(content) > max_length else content

    # Remove newlines and extra spaces for single line display
    clean_content = " ".join(truncated.split())

    if feedback_type == "FEEDBACK":
        print(f"{Colors.DIM}   💭 {clean_content}{Colors.RESET}")
    elif feedback_type == "MISSING":
        print(f"{Colors.DIM}   📋 Missing: {clean_content}{Colors.RESET}")
    elif feedback_type == "SUGGESTIONS":
        print(f"{Colors.DIM}   💡 {clean_content}{Colors.RESET}")
    else:
        print(f"{Colors.DIM}   ℹ️  {clean_content}{Colors.RESET}")


def print_retro_progress_bar(current: int, total: int, width: int = 30) -> None:
    """Prints a retro ASCII progress bar with colors."""
    filled = int(width * current / total)
    bar = f"{Colors.BRIGHT_GREEN}{'#' * filled}{Colors.DIM}{'-' * (width - filled)}{Colors.RESET}"
    percentage = int(100 * current / total)

    if percentage < 30:
        color = Colors.BRIGHT_RED
    elif percentage < 70:
        color = Colors.BRIGHT_YELLOW
    else:
        color = Colors.BRIGHT_GREEN

    print(f"[{bar}] {color}{percentage:3d}%{Colors.RESET} ({current}/{total})")

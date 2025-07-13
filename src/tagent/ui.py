import threading
import time
import sys
import textwrap
import shutil

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
        BLACK = "\033[30m"  # Strong black for contrast
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"  # Avoid pure white on light background
        BRIGHT_BLACK = "\033[90m"  # Dark gray
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


_thinking_animation = None


def start_thinking(message: str = "Thinking") -> None:
    global _thinking_animation
    stop_thinking()
    _thinking_animation = ThinkingAnimation(message)
    _thinking_animation.start()


def stop_thinking() -> None:
    global _thinking_animation
    if _thinking_animation:
        _thinking_animation.stop()
        _thinking_animation = None


def _type_line(line: str, color: str, typing_speed: float = 0.002, blink_duration: float = 0.05, blink_speed: float = 0.01) -> None:
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


def print_retro_banner(text: str, char: str = "=", width: int = 60, color: str = Colors.BRIGHT_GREEN) -> None:
    """Prints a retro-style banner with typing effect."""
    border = char * width
    padding = (width - len(text) - 2) // 2
    padded_text = " " * padding + text + " " * padding
    if len(padded_text) < width - 2:
        padded_text += " "
    _type_line(border, color)
    _type_line(f"{char}{padded_text}{char}", color)
    _type_line(border, color)


def print_retro_step(step_num: int, action: str, title: str) -> None:
    """Prints a step with retro-style typing effect."""
    action_colors = {
        "EXECUTE": Colors.BRIGHT_GREEN,
        "PLAN": Colors.BRIGHT_GREEN,
        "SUMMARIZE": Colors.BRIGHT_GREEN,
        "EVALUATE": Colors.BRIGHT_GREEN,
    }
    color = action_colors.get(action.upper(), Colors.GREEN)
    step_text = f"STEP {step_num:02d}: {action.upper()}"
    _type_line(step_text, color)
    _type_line(title, Colors.DIM + Colors.GREEN)


def print_retro_status(status: str, details: str = "") -> None:
    """Prints status messages with retro-style typing effect."""
    timestamp = f"[{__import__('time').strftime('%H:%M:%S')}]"
    status_upper = status.upper()
    if status_upper == "SUCCESS":
        symbol = "[+]"
        color = Colors.BRIGHT_GREEN
    elif status_upper == "ERROR":
        symbol = "[!]"
        color = Colors.BRIGHT_RED
    elif status_upper == "WARNING":
        symbol = "[~]"
        color = Colors.BRIGHT_YELLOW
    elif status_upper == "THINKING":
        symbol = "[*]"
        color = Colors.GREEN
    elif status_upper == "EXECUTE":
        symbol = "[>]"
        color = Colors.BRIGHT_GREEN
    elif status_upper == "PLAN":
        symbol = "[#]"
        color = Colors.BRIGHT_GREEN
    elif status_upper == "EVALUATE":
        symbol = "[?]"
        color = Colors.BRIGHT_GREEN
    elif status_upper == "SUMMARIZE":
        symbol = "[=]"
        color = Colors.BRIGHT_GREEN
    elif status_upper == "FORMAT":
        symbol = "[@]"
        color = Colors.BRIGHT_GREEN
    else:
        symbol = "[-]"
        color = Colors.GREEN
    message = f"{symbol} {timestamp} {status_upper}: {details}"
    _type_line(message, color)


def print_plan_details(content: str, max_width: int = 80) -> None:
    """Prints plan details with typing effect."""
    if not content:
        return
    clean_content = " ".join(content.split())
    if len(clean_content) <= max_width:
        _type_line(f"  * Plan: {clean_content}", Colors.DIM + Colors.GREEN)
        return
    first_line = clean_content[:max_width-10] + "..."
    _type_line(f"  * Plan: {first_line}", Colors.DIM + Colors.GREEN)
    remaining = clean_content[max_width-10:]
    words = remaining.split()
    current_line = "        "
    for word in words[:15]:
        if len(current_line + word + " ") <= max_width:
            current_line += word + " "
        else:
            if current_line.strip():
                _type_line(current_line.rstrip(), Colors.DIM + Colors.GREEN)
            current_line = "        " + word + " "
    if current_line.strip() and len(current_line) > 8:
        if len(words) > 15:
            current_line = current_line.rstrip() + "..."
        _type_line(current_line.rstrip(), Colors.DIM + Colors.GREEN)


def get_terminal_width():
    """Gets the current terminal width."""
    return shutil.get_terminal_size().columns

def print_feedback_dimmed(feedback_type: str, content: str, max_length: int = None) -> None:
    """Prints feedback in dimmed text, wrapping long lines."""
    if not content:
        return
    terminal_width = get_terminal_width()
    if max_length is None:
        max_length = terminal_width - 10  # Space for padding
    wrapped_content = textwrap.wrap(content, width=max_length)
    for line in wrapped_content:
        if feedback_type == "FEEDBACK":
            print(f"{Colors.DIM}   - {line}{Colors.RESET}")
        elif feedback_type == "MISSING":
            print(f"{Colors.DIM}   # Missing: {line}{Colors.RESET}")
        elif feedback_type == "SUGGESTIONS":
            print(f"{Colors.DIM}   * {line}{Colors.RESET}")
        else:
            print(f"{Colors.DIM}   I {line}{Colors.RESET}")


def print_retro_progress_bar(current: int, total: int, width: int = 30) -> None:
    """Prints a simple progress bar without special characters."""
    filled = int(width * current / total)
    bar = f"{Colors.BRIGHT_GREEN}{'=' * filled}{Colors.GREEN}{'-' * (width - filled)}{Colors.RESET}"
    percentage = int(100 * current / total)
    print(f"[{bar}] {Colors.BRIGHT_GREEN}{percentage:3d}%{Colors.RESET} ({current}/{total})")
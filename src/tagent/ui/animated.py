import threading
import time
import sys
import textwrap
import shutil
import random
from typing import Optional

from .interface import UIInterface, MessageType


TERMINAL_MODE = "dark"


class Colors:
    """ANSI color codes based on terminal mode."""
    if TERMINAL_MODE == "dark":
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        ITALIC = "\033[3m"
        UNDERLINE = "\033[4m"
        BLINK = "\033[5m"
        REVERSE = "\033[7m"
        STRIKETHROUGH = "\033[9m"
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
        # Matrix-style greens with gradient support
        MATRIX_GREEN = "\033[38;2;0;255;65m"
        MATRIX_DARK_GREEN = "\033[38;2;0;150;40m"
        MATRIX_BRIGHT_GREEN = "\033[38;2;100;255;100m"
        MATRIX_GLOW = "\033[38;2;150;255;150m"
        # Additional gradient colors
        MATRIX_GRADIENT_1 = "\033[38;2;0;180;50m"
        MATRIX_GRADIENT_2 = "\033[38;2;50;200;75m"
        MATRIX_GRADIENT_3 = "\033[38;2;80;220;90m"
        MATRIX_GRADIENT_4 = "\033[38;2;120;240;120m"
    else:  # Light mode
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        ITALIC = "\033[3m"
        UNDERLINE = "\033[4m"
        BLINK = "\033[5m"
        REVERSE = "\033[7m"
        STRIKETHROUGH = "\033[9m"
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
        # Matrix-style greens (adapted for light mode)
        MATRIX_GREEN = "\033[38;2;0;120;30m"
        MATRIX_DARK_GREEN = "\033[38;2;0;80;20m"
        MATRIX_BRIGHT_GREEN = "\033[38;2;0;160;50m"
        MATRIX_GLOW = "\033[38;2;0;200;60m"
        # Additional gradient colors for light mode
        MATRIX_GRADIENT_1 = "\033[38;2;0;100;25m"
        MATRIX_GRADIENT_2 = "\033[38;2;0;130;35m"
        MATRIX_GRADIENT_3 = "\033[38;2;0;140;40m"
        MATRIX_GRADIENT_4 = "\033[38;2;0;180;55m"


class MatrixEffects:
    """Matrix-style terminal effects."""
    
    @staticmethod
    def get_matrix_char() -> str:
        """Get a random Matrix-style character."""
        # Japanese katakana and ASCII characters for authentic Matrix feel
        chars = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
        ascii_chars = "01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Weight towards digital characters
        all_chars = chars + ascii_chars * 2 + symbols
        return random.choice(all_chars)
    
    @staticmethod
    def create_scanline(width: int, intensity: float = 0.3) -> str:
        """Create a scanline effect."""
        if random.random() > intensity:
            return ""
        line = ""
        for i in range(width):
            if random.random() < 0.1:
                line += Colors.MATRIX_GLOW + MatrixEffects.get_matrix_char() + Colors.RESET
            else:
                line += " "
        return line
    
    @staticmethod
    def glitch_text(text: str, intensity: float = 0.05) -> str:
        """Add glitch effect to text."""
        if random.random() > intensity:
            return text
        
        result = ""
        for char in text:
            if random.random() < 0.1:
                # Character substitution
                result += Colors.MATRIX_GLOW + MatrixEffects.get_matrix_char() + Colors.RESET
            elif random.random() < 0.05:
                # Character duplication
                result += char + char
            else:
                result += char
        return result
    
    @staticmethod
    def create_data_stream(width: int, length: int = 3) -> str:
        """Create a vertical data stream effect."""
        stream = ""
        for i in range(length):
            intensity = 1.0 - (i / length)
            if intensity > 0.7:
                color = Colors.MATRIX_BRIGHT_GREEN
            elif intensity > 0.4:
                color = Colors.MATRIX_GREEN
            else:
                color = Colors.MATRIX_DARK_GREEN
            
            char = MatrixEffects.get_matrix_char()
            stream += f"{color}{char}{Colors.RESET}"
        return stream
    
    @staticmethod
    def create_gradient_text(text: str, start_color: str = None, end_color: str = None) -> str:
        """Create gradient text effect for Matrix elements."""
        if not text or len(text) < 2:
            return text
        
        # Default gradient colors
        gradient_colors = [
            Colors.MATRIX_DARK_GREEN,
            Colors.MATRIX_GRADIENT_1,
            Colors.MATRIX_GRADIENT_2,
            Colors.MATRIX_GREEN,
            Colors.MATRIX_GRADIENT_3,
            Colors.MATRIX_GRADIENT_4,
            Colors.MATRIX_BRIGHT_GREEN,
            Colors.MATRIX_GLOW
        ]
        
        if start_color and end_color:
            gradient_colors = [start_color, end_color]
        
        result = ""
        text_length = len(text)
        
        for i, char in enumerate(text):
            # Calculate position in gradient (0.0 to 1.0)
            position = i / (text_length - 1) if text_length > 1 else 0
            
            # Select color based on position
            color_index = int(position * (len(gradient_colors) - 1))
            color = gradient_colors[min(color_index, len(gradient_colors) - 1)]
            
            result += f"{color}{char}"
        
        return result + Colors.RESET


class ThinkingAnimation:
    """Matrix-style thinking animation with vertical cascading Japanese characters."""
    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.running = False
        self.thread = None
        self.terminal_width = shutil.get_terminal_size().columns
        self.cascade_columns = []  # Each column tracks its cascade state
        self.num_columns = 20  # Number of cascade columns
        self.cascade_height = 4  # 4 stages of gradient trail
        self.animation_lines = []  # Buffer for multi-line animation

    def start(self):
        if not self.running:
            self.running = True
            # Initialize cascade columns
            self.cascade_columns = []
            for i in range(self.num_columns):
                self.cascade_columns.append({
                    'chars': [' '] * self.cascade_height,  # Trail of characters
                    'intensities': [0] * self.cascade_height,  # Fade levels
                    'active': random.random() < 0.3,  # Start some columns active
                    'delay': random.randint(0, 10)  # Random start delay
                })
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=0.5)
            # Clear all animation lines and show final message
            self._clear_animation_and_show_final()

    def _clear_animation_and_show_final(self):
        """Clear all animation lines and show final message."""
        # Clear animation lines by moving cursor up and clearing
        if self.animation_lines:
            for i in range(len(self.animation_lines)):
                sys.stdout.write(f"\033[A\033[K")  # Move up and clear line
            sys.stdout.flush()
        
        # Show final message
        symbol_color = Colors.MATRIX_BRIGHT_GREEN
        timestamp = f"[{__import__('time').strftime('%H:%M:%S')}]"
        
        final_message = f"{symbol_color}⟨*⟩{Colors.RESET} {symbol_color}{timestamp}{Colors.RESET} {symbol_color}{Colors.BOLD}{self.message.upper()}{Colors.RESET}"
        sys.stdout.write(final_message + '\n')
        sys.stdout.flush()

    def _get_matrix_char(self) -> str:
        """Get Japanese Matrix character."""
        japanese_chars = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
        return random.choice(japanese_chars)

    def _get_fade_color(self, intensity: int) -> str:
        """Get color based on fade intensity (0-4)."""
        if intensity == 4:
            return Colors.MATRIX_BRIGHT_GREEN  # Brightest
        elif intensity == 3:
            return Colors.MATRIX_GREEN  # Medium-bright
        elif intensity == 2:
            return Colors.MATRIX_GRADIENT_1  # Medium
        elif intensity == 1:
            return Colors.MATRIX_DARK_GREEN  # Fading
        else:
            return Colors.BRIGHT_BLACK  # Almost gone

    def _update_cascade_column(self, column: dict):
        """Update a single cascade column."""
        if column['delay'] > 0:
            column['delay'] -= 1
            return
        
        if not column['active']:
            # Random chance to activate
            if random.random() < 0.1:
                column['active'] = True
                column['chars'][0] = self._get_matrix_char()
                column['intensities'][0] = 4
            return
        
        # Shift characters down (cascade effect)
        for i in range(self.cascade_height - 1, 0, -1):
            column['chars'][i] = column['chars'][i-1]
            column['intensities'][i] = column['intensities'][i-1] - 1 if column['intensities'][i-1] > 0 else 0
        
        # Add new character at top or fade out
        if random.random() < 0.7:  # 70% chance to continue
            column['chars'][0] = self._get_matrix_char()
            column['intensities'][0] = 4
        else:
            column['chars'][0] = ' '
            column['intensities'][0] = 0
            # Random chance to deactivate
            if random.random() < 0.3:
                column['active'] = False

    def _render_cascade_frame(self) -> list:
        """Render a single frame of the cascade animation."""
        frame_lines = []
        
        # Create cascade_height lines
        for row in range(self.cascade_height):
            line = ""
            for col in range(self.num_columns):
                if col < len(self.cascade_columns):
                    char = self.cascade_columns[col]['chars'][row]
                    intensity = self.cascade_columns[col]['intensities'][row]
                    
                    if intensity > 0:
                        color = self._get_fade_color(intensity)
                        line += f"{color}{char}{Colors.RESET}"
                    else:
                        line += " "
                else:
                    line += " "
            
            # Add message info on first line
            if row == 0:
                line += f"  {Colors.MATRIX_GLOW}⟨*⟩ {self.message}...{Colors.RESET}"
            
            frame_lines.append(line)
        
        return frame_lines

    def _animate(self):
        frame_count = 0
        
        while self.running:
            # Update all cascade columns
            for column in self.cascade_columns:
                self._update_cascade_column(column)
            
            # Render frame
            frame_lines = self._render_cascade_frame()
            
            # Clear previous frame if exists
            if self.animation_lines:
                for i in range(len(self.animation_lines)):
                    sys.stdout.write(f"\033[A\033[K")  # Move up and clear line
            
            # Display new frame
            for line in frame_lines:
                sys.stdout.write(line + '\n')
            
            sys.stdout.flush()
            self.animation_lines = frame_lines
            
            time.sleep(0.1)  # Animation speed
            frame_count += 1


class AnimatedUI(UIInterface):
    """Matrix-style animated UI implementation with advanced terminal effects."""
    
    def __init__(self):
        self._thinking_animation = None
        self._terminal_width = shutil.get_terminal_size().columns
        self._enable_effects = True  # Can be disabled for performance
        self._effect_intensity = 0.4  # Moderate default intensity for nice effects
    
    def _get_color_for_message_type(self, message_type: MessageType) -> str:
        """Map semantic message types to Matrix-style colors."""
        color_map = {
            MessageType.SUCCESS: Colors.MATRIX_BRIGHT_GREEN,
            MessageType.ERROR: Colors.BRIGHT_RED,
            MessageType.WARNING: Colors.BRIGHT_YELLOW,
            MessageType.INFO: Colors.BRIGHT_CYAN,
            MessageType.THINKING: Colors.MATRIX_GREEN,
            MessageType.EXECUTE: Colors.MATRIX_BRIGHT_GREEN,
            MessageType.PLAN: Colors.MATRIX_GLOW,
            MessageType.EVALUATE: Colors.MATRIX_GREEN,
            MessageType.SUMMARIZE: Colors.MATRIX_BRIGHT_GREEN,
            MessageType.FORMAT: Colors.MATRIX_GREEN,
            MessageType.PRIMARY: Colors.MATRIX_BRIGHT_GREEN,
            MessageType.SECONDARY: Colors.MATRIX_GREEN,
            MessageType.MUTED: Colors.MATRIX_DARK_GREEN,
        }
        return color_map.get(message_type, Colors.MATRIX_GREEN)
    
    def _get_symbol_color_for_message_type(self, message_type: MessageType) -> str:
        """Get the appropriate color for status symbols and prefixes."""
        symbol_color_map = {
            MessageType.SUCCESS: Colors.MATRIX_BRIGHT_GREEN,
            MessageType.ERROR: Colors.BRIGHT_RED,
            MessageType.WARNING: Colors.BRIGHT_YELLOW,
            MessageType.INFO: Colors.BRIGHT_CYAN,
            MessageType.THINKING: Colors.MATRIX_GREEN,
            MessageType.EXECUTE: Colors.MATRIX_BRIGHT_GREEN,
            MessageType.PLAN: Colors.MATRIX_GLOW,
            MessageType.EVALUATE: Colors.MATRIX_GREEN,
            MessageType.SUMMARIZE: Colors.MATRIX_BRIGHT_GREEN,
            MessageType.FORMAT: Colors.MATRIX_GREEN,
            MessageType.PRIMARY: Colors.MATRIX_BRIGHT_GREEN,
            MessageType.SECONDARY: Colors.MATRIX_GREEN,
            MessageType.MUTED: Colors.MATRIX_DARK_GREEN,
        }
        return symbol_color_map.get(message_type, Colors.MATRIX_GREEN)
    
    def set_effect_intensity(self, intensity: float) -> None:
        """Set the intensity of Matrix effects (0.0 to 1.0)."""
        self._effect_intensity = max(0.0, min(1.0, intensity))
        self._enable_effects = intensity > 0.0
    
    def disable_effects(self) -> None:
        """Disable all Matrix effects for better performance."""
        self._enable_effects = False
        self._effect_intensity = 0.0
    
    def start_thinking(self, message: str = "Thinking") -> None:
        """Start thinking animation."""
        self.stop_thinking()
        self._thinking_animation = ThinkingAnimation(message)
        self._thinking_animation.start()
    
    def stop_thinking(self) -> None:
        """Stop thinking animation with clean transition."""
        if self._thinking_animation:
            self._thinking_animation.stop()
            self._thinking_animation = None
            # Add a small delay to ensure clean transition
            time.sleep(0.1)
    
    def _type_line(self, line: str, color: str, typing_speed: float = 0.001, blink_duration: float = 0.08, blink_speed: float = 0.02) -> None:
        """Writes a line with Matrix-style typing effect and advanced cursor."""
        sys.stdout.write(color)
        
        # Add occasional glitch effect during typing (very minimal)
        if self._enable_effects and random.random() < 0.01 * self._effect_intensity:
            line = MatrixEffects.glitch_text(line, 0.01)
        
        for i, char in enumerate(line):
            # Random slight delay variation for more organic feel
            delay = typing_speed + random.uniform(-0.0005, 0.0005)
            
            # Occasionally show Matrix character before correct character (very rare)
            if self._enable_effects and random.random() < 0.005 * self._effect_intensity:
                matrix_char = MatrixEffects.get_matrix_char()
                sys.stdout.write(Colors.MATRIX_GLOW + matrix_char)
                sys.stdout.flush()
                time.sleep(delay)
                sys.stdout.write('\b \b')
            
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        
        # Simple cursor blink for clean output
        cursor_char = '|'
        num_blinks = int(blink_duration / (2 * blink_speed))
        
        for i in range(num_blinks):
            sys.stdout.write(f'{Colors.MATRIX_GLOW}{cursor_char}')
            sys.stdout.flush()
            time.sleep(blink_speed)
            sys.stdout.write('\b \b')
            sys.stdout.flush()
            time.sleep(blink_speed)
        
        sys.stdout.write(Colors.RESET + '\n')
        
        # Optional scanline effect after line (disabled for clean output)
        # if self._enable_effects and random.random() < 0.01 * self._effect_intensity:
        #     scanline = MatrixEffects.create_scanline(self._terminal_width, 0.01)
        #     if scanline:
        #         sys.stdout.write(scanline + '\n')
        #         sys.stdout.flush()
    
    def print_banner(self, text: str, char: str = "=", width: int = 60, message_type: MessageType = MessageType.PRIMARY) -> None:
        """Prints an epic Matrix-style banner with advanced ASCII art."""
        color = self._get_color_for_message_type(message_type)
        
        # Advanced ASCII art banners based on message type
        if message_type == MessageType.PRIMARY:
            self._print_matrix_primary_banner(text, width)
        elif message_type == MessageType.SUCCESS:
            self._print_matrix_success_banner(text, width)
        elif message_type == MessageType.WARNING:
            self._print_matrix_warning_banner(text, width)
        else:
            # Fallback to enhanced border
            self._print_enhanced_border_banner(text, char, width, color)
    
    def _print_matrix_primary_banner(self, text: str, width: int) -> None:
        """Print advanced Matrix-style primary banner."""
        # Top border with complex ASCII
        top_border = "╔" + "═" * (width - 2) + "╗"
        if self._enable_effects:
            top_border = MatrixEffects.create_gradient_text(top_border)
        sys.stdout.write(top_border + '\n')
        
        # Secondary decorative line
        deco_line = "║" + "▓▒░" * ((width - 2) // 3) + "▓" * ((width - 2) % 3) + "║"
        if self._enable_effects:
            deco_line = MatrixEffects.create_gradient_text(deco_line)
        sys.stdout.write(deco_line + '\n')
        
        # Main text line
        padding = (width - len(text) - 2) // 2
        padded_text = " " * padding + text + " " * padding
        if len(padded_text) < width - 2:
            padded_text += " "
        main_line = f"║{padded_text}║"
        sys.stdout.write(f"{Colors.MATRIX_BRIGHT_GREEN}{Colors.BOLD}{main_line}{Colors.RESET}\n")
        
        # Bottom decorative line
        sys.stdout.write(deco_line + '\n')
        
        # Bottom border
        bottom_border = "╚" + "═" * (width - 2) + "╝"
        if self._enable_effects:
            bottom_border = MatrixEffects.create_gradient_text(bottom_border)
        sys.stdout.write(bottom_border + '\n')
        
        sys.stdout.flush()
    
    def _print_matrix_success_banner(self, text: str, width: int) -> None:
        """Print advanced Matrix-style success banner."""
        # Diamond pattern top
        diamond_top = "◆" + "◇" * (width - 2) + "◆"
        if self._enable_effects:
            diamond_top = MatrixEffects.create_gradient_text(diamond_top)
        sys.stdout.write(diamond_top + '\n')
        
        # Success indicators
        success_line = "◇" + "★" * ((width - 2) // 2) + "☆" * ((width - 2) // 2) + "◇"
        if self._enable_effects:
            success_line = MatrixEffects.create_gradient_text(success_line)
        sys.stdout.write(success_line + '\n')
        
        # Main text
        padding = (width - len(text) - 2) // 2
        padded_text = " " * padding + text + " " * padding
        if len(padded_text) < width - 2:
            padded_text += " "
        main_line = f"◇{padded_text}◇"
        sys.stdout.write(f"{Colors.MATRIX_BRIGHT_GREEN}{Colors.BOLD}{main_line}{Colors.RESET}\n")
        
        # Bottom success line
        sys.stdout.write(success_line + '\n')
        
        # Diamond pattern bottom
        diamond_bottom = "◆" + "◇" * (width - 2) + "◆"
        if self._enable_effects:
            diamond_bottom = MatrixEffects.create_gradient_text(diamond_bottom)
        sys.stdout.write(diamond_bottom + '\n')
        
        sys.stdout.flush()
    
    def _print_matrix_warning_banner(self, text: str, width: int) -> None:
        """Print advanced Matrix-style warning banner."""
        # Warning zigzag pattern
        zigzag = "⚠" + "▲▼" * ((width - 2) // 2) + "⚠"
        if len(zigzag) < width:
            zigzag += "▲" * (width - len(zigzag))
        sys.stdout.write(f"{Colors.BRIGHT_YELLOW}{zigzag}{Colors.RESET}\n")
        
        # Alert line
        alert_line = "▲" + "░▒▓" * ((width - 2) // 3) + "▲"
        if len(alert_line) < width:
            alert_line += "░" * (width - len(alert_line))
        sys.stdout.write(f"{Colors.BRIGHT_YELLOW}{alert_line}{Colors.RESET}\n")
        
        # Main text
        padding = (width - len(text) - 2) // 2
        padded_text = " " * padding + text + " " * padding
        if len(padded_text) < width - 2:
            padded_text += " "
        main_line = f"▲{padded_text}▲"
        sys.stdout.write(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}{main_line}{Colors.RESET}\n")
        
        # Bottom alert and zigzag
        sys.stdout.write(f"{Colors.BRIGHT_YELLOW}{alert_line}{Colors.RESET}\n")
        sys.stdout.write(f"{Colors.BRIGHT_YELLOW}{zigzag}{Colors.RESET}\n")
        
        sys.stdout.flush()
    
    def _print_enhanced_border_banner(self, text: str, char: str, width: int, color: str) -> None:
        """Print enhanced border banner for other message types."""
        # Enhanced border characters
        border_chars = {
            "=": "═━▬─",
            "*": "★☆✦✧",
            "#": "▓▒░█",
            "-": "─═━▬"
        }
        
        chars = border_chars.get(char, char)
        border = ''.join(random.choice(chars) if random.random() < 0.4 else char for _ in range(width))
        
        # Apply gradient if effects enabled
        if self._enable_effects:
            border = MatrixEffects.create_gradient_text(border)
        
        # Print enhanced banner
        sys.stdout.write(border + '\n')
        
        # Main text
        padding = (width - len(text) - 2) // 2
        padded_text = " " * padding + text + " " * padding
        if len(padded_text) < width - 2:
            padded_text += " "
        main_line = f"{char}{padded_text}{char}"
        sys.stdout.write(f"{color}{Colors.BOLD}{main_line}{Colors.RESET}\n")
        
        # Bottom border
        sys.stdout.write(border + '\n')
        sys.stdout.flush()
    
    def _create_data_rain(self, width: int, lines: int) -> None:
        """Create a subtle data rain effect above/below banners."""
        for _ in range(lines):
            rain_line = ""
            for i in range(width):
                if random.random() < 0.08 * self._effect_intensity:
                    char = MatrixEffects.get_matrix_char()
                    intensity = random.choice([Colors.MATRIX_DARK_GREEN, Colors.MATRIX_GREEN])
                    rain_line += f"{intensity}{char}{Colors.RESET}"
                else:
                    rain_line += " "
            
            sys.stdout.write(rain_line + '\n')
            sys.stdout.flush()
            time.sleep(0.03)
    
    def print_step(self, step_num: int, action: str, title: str) -> None:
        """Prints a step with proper color formatting."""
        action_message_types = {
            "EXECUTE": MessageType.EXECUTE,
            "PLAN": MessageType.PLAN,
            "SUMMARIZE": MessageType.SUMMARIZE,
            "EVALUATE": MessageType.EVALUATE,
        }
        message_type = action_message_types.get(action.upper(), MessageType.PRIMARY)
        step_color = self._get_color_for_message_type(message_type)
        
        # Create step text with proper coloring
        step_prefix = f"STEP {step_num:02d}"
        action_display = action.upper()
        
        # Build step message with consistent colors
        step_message = f"{step_color}{step_prefix}: {Colors.BOLD}{action_display}{Colors.RESET}"
        
        # Type the step message
        sys.stdout.write(step_message)
        sys.stdout.write('\n')
        sys.stdout.flush()
        
        # Title with muted Matrix green
        title_color = Colors.MATRIX_DARK_GREEN
        title_message = f"{title_color}    {title}{Colors.RESET}"
        
        # Type the title
        sys.stdout.write(title_message)
        sys.stdout.write('\n')
        sys.stdout.flush()
    
    def print_status(self, status: str, details: str = "", message_type: Optional[MessageType] = None) -> None:
        """Prints status messages with proper color formatting."""
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
        
        # Enhanced status symbols with Matrix-style brackets
        status_symbols = {
            MessageType.SUCCESS: "⟨+⟩",
            MessageType.ERROR: "⟨!⟩",
            MessageType.WARNING: "⟨~⟩",
            MessageType.THINKING: "⟨*⟩",
            MessageType.EXECUTE: "⟨>⟩",
            MessageType.PLAN: "⟨#⟩",
            MessageType.EVALUATE: "⟨?⟩",
            MessageType.SUMMARIZE: "⟨=⟩",
            MessageType.FORMAT: "⟨@⟩",
        }
        
        symbol = status_symbols.get(message_type, "⟨-⟩")
        symbol_color = self._get_symbol_color_for_message_type(message_type)
        
        # Build message with proper color separation
        # Symbol and timestamp with semantic color
        prefix = f"{symbol_color}{symbol}{Colors.RESET} {symbol_color}{timestamp}{Colors.RESET}"
        
        # Status text with semantic color and bold
        status_text = f"{symbol_color}{Colors.BOLD}{status_upper}{Colors.RESET}"
        
        # Details with Matrix green (or white for better readability)
        details_color = Colors.MATRIX_GREEN if message_type in [MessageType.THINKING, MessageType.EXECUTE, MessageType.PLAN, MessageType.EVALUATE, MessageType.SUMMARIZE, MessageType.FORMAT] else Colors.WHITE
        details_text = f"{details_color}{details}{Colors.RESET}" if details else ""
        
        # Combine all parts
        if details_text:
            message = f"{prefix} {status_text}: {details_text}"
        else:
            message = f"{prefix} {status_text}"
        
        # Type the message without additional coloring (colors are already applied)
        sys.stdout.write(message)
        sys.stdout.write('\n')
        sys.stdout.flush()
    
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
        """Prints a Matrix-style progress bar with advanced effects."""
        filled = int(width * current / total)
        success_color = self._get_color_for_message_type(MessageType.SUCCESS)
        secondary_color = self._get_color_for_message_type(MessageType.SECONDARY)
        percentage = int(100 * current / total)
        
        if self._enable_effects:
            # Create Matrix-style progress bar with varied characters
            filled_chars = "█▉▊▋▌▍▎▏"
            empty_chars = "░▒▓"
            
            bar_content = ""
            for i in range(width):
                if i < filled:
                    # Use different characters for visual variety
                    if i == filled - 1 and filled < width:
                        # Partial fill character
                        char = random.choice(filled_chars[1:])
                    else:
                        char = random.choice(filled_chars[:3])
                    bar_content += f"{success_color}{char}"
                else:
                    char = random.choice(empty_chars)
                    bar_content += f"{secondary_color}{char}"
            
            bar_content += Colors.RESET
            
            # Add Matrix-style brackets and data streams
            left_stream = MatrixEffects.create_data_stream(2, 1)
            right_stream = MatrixEffects.create_data_stream(2, 1)
            
            # Progress percentage with gradient effect
            percentage_text = f"{percentage:3d}%"
            if self._effect_intensity > 0.5:
                percentage_display = MatrixEffects.create_gradient_text(percentage_text)
            else:
                percentage_display = f"{Colors.MATRIX_GLOW}{percentage_text}{Colors.RESET}"
            
            progress_line = f"{left_stream}⟨{bar_content}⟩{right_stream} {percentage_display} ({current}/{total})"
            
        else:
            # Simple fallback
            bar = f"{success_color}{'=' * filled}{secondary_color}{'-' * (width - filled)}{Colors.RESET}"
            progress_line = f"[{bar}] {success_color}{percentage:3d}%{Colors.RESET} ({current}/{total})"
        
        print(progress_line)
"""
Utility functions for TAgent.
"""

from typing import Dict, Any, Callable, List
import inspect


def get_tool_documentation(tools: Dict[str, Callable]) -> str:
    """
    Extracts documentation from registered tools including docstrings and signatures.

    Args:
        tools: Dictionary of registered tools

    Returns:
        Formatted string with tool documentation
    """
    if not tools:
        return ""

    tool_docs = []

    for tool_name, tool_func in tools.items():
        # Extract function signature
        try:
            sig = inspect.signature(tool_func)
            signature = f"{tool_name}{sig}"
        except (ValueError, TypeError):
            signature = f"{tool_name}(state, args)"

        # Extract docstring
        docstring = inspect.getdoc(tool_func)
        if not docstring:
            docstring = "No documentation available"

        tool_doc = f"- {signature}: {docstring}"
        tool_docs.append(tool_doc)

    return "Available tools:\n" + "\n".join(tool_docs) + "\n"


def detect_action_loop(recent_actions: List[str], max_recent: int = 3) -> bool:
    """Detects if agent is in a loop of repeated actions."""
    if len(recent_actions) < 2:
        return False

    # Detect 'evaluate' loops early (even with 2 repetitions)
    if recent_actions[-1] == "evaluate" and recent_actions[-2] == "evaluate":
        return True

    # Original check for 3+ actions
    if len(recent_actions) >= max_recent:
        last_actions = recent_actions[-max_recent:]
        return len(set(last_actions)) == 1

    return False


def format_conversation_as_chat(conversation_history: List[Dict[str, str]]) -> str:
    """
    Formats conversation history as readable chat.

    Args:
        conversation_history: List of conversation messages

    Returns:
        String formatted as chat
    """
    chat_lines = []
    chat_lines.append("=== CONVERSATION HISTORY ===\n")

    for i, message in enumerate(conversation_history, 1):
        role = message.get("role", "unknown")
        content = message.get("content", "")

        if role == "user":
            chat_lines.append(f"👤 USER [{i}]:")
            chat_lines.append(f"   {content}\n")
        elif role == "assistant":
            chat_lines.append(f"🤖 ASSISTANT [{i}]:")
            chat_lines.append(f"   {content}\n")
        else:
            chat_lines.append(f"📝 {role.upper()} [{i}]:")
            chat_lines.append(f"   {content}\n")

    chat_lines.append("=== END OF HISTORY ===")
    return "\n".join(chat_lines)

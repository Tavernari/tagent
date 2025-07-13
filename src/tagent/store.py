"""
Redux-inspired state management for TAgent.
"""

from typing import Dict, Any, Callable, Optional, Tuple, List
from pydantic import BaseModel

from .models import AgentState, StructuredResponse


class Store:
    """Redux-inspired store for managing agent state and conversation history."""

    def __init__(self, initial_state: Dict[str, Any]):
        self.state = AgentState(data=initial_state)
        self.tools: Dict[str, Callable] = {}  # Registry of custom tools
        self.conversation_history: List[Dict[str, str]] = []  # Conversation history

    def register_tool(
        self,
        name: str,
        tool_func: Callable[
            [Dict[str, Any], Dict[str, Any]], Optional[Tuple[str, BaseModel]]
        ],
    ):
        """Registers a custom tool as an action."""
        self.tools[name] = tool_func

    def add_to_conversation(self, role: str, content: str) -> None:
        """Adds message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def add_assistant_response(self, response: StructuredResponse) -> None:
        """Adds assistant response to history in structured format."""
        formatted_response = f"Action: {response.action}\nReasoning: {response.reasoning}\nParams: {response.params}"
        self.add_to_conversation("assistant", formatted_response)

    def dispatch(
        self,
        action_func: Callable[[Dict[str, Any]], Optional[Tuple[str, BaseModel]]],
        verbose: bool = False,
    ) -> None:
        """Dispatches an action: calls function, applies reducer."""
        if verbose:
            print("[INFO] Dispatching action...")
        result = action_func(self.state.data)
        if result:
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, tuple) and len(item) == 2:
                        key, value = item
                        self.state.data[key] = value
            elif (
                isinstance(result, tuple) and len(result) == 2
            ):  # Correção: len(result) em vez de len(item)
                key, value = result
                self.state.data[key] = value
        if verbose:
            print(f"[LOG] State updated: {self.state.data}")

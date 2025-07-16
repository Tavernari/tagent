"""
Smart Tool Executor for TAgent

This module provides a flexible and intelligent tool execution engine that
adapts to the tool's signature, rather than forcing the tool to adapt to
a rigid framework.
"""

import inspect
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, Callable, Optional, Tuple

class ToolExecutor:
    """
    A class that intelligently executes tools by inspecting their signatures
    and adapting the calling convention accordingly.
    """

    def execute(
        self,
        tool_func: Callable,
        tool_name: str,
        agent_state: Dict[str, Any],
        llm_args: Dict[str, Any],
    ) -> Optional[Tuple[str, Any]]:
        """
        Executes a tool by dynamically matching its signature with the
        available state and arguments.

        Args:
            tool_func: The callable tool function to execute.
            tool_name: The name of the tool.
            agent_state: The current state of the agent.
            llm_args: The arguments for the tool provided by the LLM.

        Returns:
            A tuple `(key, value)` for updating the agent's state, or None.
        """
        try:
            sig = inspect.signature(tool_func)
            call_args = {}
            pydantic_model_param = None
            
            # 1. Inspect parameters to build the call arguments
            for param in sig.parameters.values():
                if inspect.isclass(param.annotation) and issubclass(param.annotation, BaseModel) and param.annotation is not BaseModel:
                    # Found the Pydantic model parameter
                    if pydantic_model_param is not None:
                        raise TypeError(f"Tool '{tool_name}' has multiple Pydantic BaseModel parameters. Only one is allowed.")
                    pydantic_model_param = param
                elif param.name == "state" and param.annotation in (Dict, Dict[str, Any], Any):
                    # Found the state parameter
                    call_args[param.name] = agent_state
                elif param.name in llm_args:
                    # Found a simple parameter (str, int, etc.)
                    call_args[param.name] = llm_args[param.name]

            # 2. Validate and instantiate the Pydantic model if present
            if pydantic_model_param:
                try:
                    model_instance = pydantic_model_param.annotation(**llm_args)
                    call_args[pydantic_model_param.name] = model_instance
                except ValidationError as e:
                    error_msg = f"Argument validation failed for tool '{tool_name}': {e}"
                    return ("tool_error", {"tool": tool_name, "error": error_msg})

            # 3. Execute the tool
            result = tool_func(**call_args)

            # 4. Normalize the output for state update
            return_annotation = sig.return_annotation
            
            if result is None or return_annotation is None:
                # Fire-and-forget tool
                return None
            
            if isinstance(result, BaseModel):
                # Return type is a Pydantic model, infer key from class name
                key = result.__class__.__name__.lower()
                return (key, result)

            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
                # Traditional (key, value) tuple for explicit state update
                return result

            # For basic return types (str, list, dict), infer key from tool name
            return (f"{tool_name}_output", result)

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {e}"
            return ("tool_error", {"tool": tool_name, "error": error_msg}) 
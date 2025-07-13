"""
LLM Adapter for TAgent - provides a clean interface for LLM interactions that's easy to mock.
"""

from typing import Dict, Optional, List, Any
from abc import ABC, abstractmethod
import json
import litellm

from .models import StructuredResponse


class LLMResponse:
    """Wrapper for LLM response data."""
    
    def __init__(self, content: str, model: str = "unknown"):
        self.content = content
        self.model = model


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        api_key: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """Complete a chat conversation and return response."""
        pass
    
    @abstractmethod
    def get_supported_params(self, model: str) -> List[str]:
        """Get list of supported parameters for a model."""
        pass


class LiteLLMAdapter(LLMAdapter):
    """LiteLLM implementation of the LLM adapter."""
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        api_key: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """Complete using LiteLLM."""
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                api_key=api_key,
                **kwargs
            )
            content = response.choices[0].message.content.strip()
            return LLMResponse(content=content, model=model)
        except Exception as e:
            raise ValueError(f"LiteLLM completion failed: {str(e)}")
    
    def get_supported_params(self, model: str) -> List[str]:
        """Get supported parameters from LiteLLM."""
        try:
            supported_params = litellm.get_supported_openai_params(model=model)
            return supported_params if supported_params else []
        except Exception:
            return []


class MockLLMAdapter(LLMAdapter):
    """Mock implementation for testing."""
    
    def __init__(self, responses: List[str] = None):
        self.responses = responses or []
        self.call_count = 0
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        api_key: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """Return pre-configured mock responses."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return LLMResponse(content=response, model=model)
        else:
            # Default response if no more responses configured
            default_response = '{"action": "evaluate", "params": {"achieved": true}, "reasoning": "Mock response"}'
            return LLMResponse(content=default_response, model=model)
    
    def get_supported_params(self, model: str) -> List[str]:
        """Return mock supported parameters."""
        return ["response_format", "temperature", "max_tokens"]
    
    def reset(self):
        """Reset call count for reuse."""
        self.call_count = 0


# Global adapter instance - can be swapped for testing
_llm_adapter: LLMAdapter = LiteLLMAdapter()


def set_llm_adapter(adapter: LLMAdapter):
    """Set the global LLM adapter (useful for testing)."""
    global _llm_adapter
    _llm_adapter = adapter


def get_llm_adapter() -> LLMAdapter:
    """Get the current LLM adapter."""
    return _llm_adapter


def parse_structured_response(
    json_str: str, 
    verbose: bool = False
) -> StructuredResponse:
    """
    Parse a JSON string into a StructuredResponse with error handling.
    
    Args:
        json_str: JSON string to parse
        verbose: Enable verbose logging
        
    Returns:
        StructuredResponse object
        
    Raises:
        ValueError: If parsing fails after all attempts
    """
    if verbose:
        print(f"[RESPONSE] Raw LLM output: {json_str}")

    # Attempt to fix common JSON formatting issues
    try:
        # Try parsing as-is first
        return StructuredResponse.model_validate_json(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        if verbose:
            print(f"[ERROR] Initial JSON parsing failed: {e}")
        
        # Try fixing single quotes to double quotes as a fallback
        try:
            import re
            # Replace single quotes with double quotes for JSON keys and string values
            fixed_json = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Fix keys
            fixed_json = re.sub(r": '([^']*)'", r': "\1"', fixed_json)  # Fix string values
            if verbose:
                print(f"[RESPONSE] Attempting with fixed quotes: {fixed_json}")
            return StructuredResponse.model_validate_json(fixed_json)
        except (ValueError, json.JSONDecodeError):
            # Try parsing as raw JSON and wrapping in correct structure
            try:
                parsed = json.loads(json_str)
                
                # If it's already a dict but missing required fields, try to fix it
                if isinstance(parsed, dict):
                    if "action" not in parsed:
                        # Wrap the response in the expected structure
                        wrapped_response = {
                            "action": "summarize",  # Default action
                            "params": parsed,
                            "reasoning": "Auto-wrapped response from LLM"
                        }
                        if verbose:
                            print(f"[RESPONSE] Wrapping response: {json.dumps(wrapped_response)}")
                        return StructuredResponse.model_validate(wrapped_response)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # If all attempts fail, raise error
            raise ValueError(f"Failed to parse JSON response: {json_str}")


def query_llm_with_adapter(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    max_retries: int = 3,
    tools: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    verbose: bool = False,
) -> StructuredResponse:
    """
    Query LLM using the adapter pattern for easier testing.
    
    Args:
        prompt: The prompt to send
        model: Model name
        api_key: API key for the LLM service
        max_retries: Maximum number of retries
        tools: Available tools dictionary
        conversation_history: Previous conversation messages
        verbose: Enable verbose logging
        
    Returns:
        StructuredResponse with structured output
        
    Raises:
        ValueError: If no valid response after retries
    """
    from .utils import get_tool_documentation
    
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant designed to output JSON. For 'evaluate' actions, include specific feedback. Example: {\"action\": \"execute\", \"params\": {\"tool\": \"tool_name\", \"args\": {\"parameter\": \"value\"}}, \"reasoning\": \"Reason to execute the action.\"} or {\"action\": \"evaluate\", \"params\": {\"achieved\": false, \"missing_items\": [\"item1\", \"item2\"], \"suggestions\": [\"suggestion1\"]}, \"reasoning\": \"Detailed explanation of what is missing and why goal is not achieved.\"}",
    }

    # Use detailed tool documentation if available
    available_tools = ""
    if tools:
        available_tools = get_tool_documentation(tools)

    user_message = {
        "role": "user",
        "content": (
            f"{prompt}\n\n"
            f"{available_tools}"
            "For 'execute' actions, consider selecting appropriate tools based on their documentation. "
            "Include 'tool' (tool name) and 'args' (parameters) in params.\n"
            "For 'evaluate' actions where the goal is not achieved, consider including 'missing_items' and 'suggestions' in params.\n"
            "Please respond with valid JSON in the format: "
            "{\"action\": str (plan|execute|summarize|evaluate), \"params\": dict, \"reasoning\": str}."
            "Use double quotes for JSON strings."
        ),
    }

    # Build messages including conversation history
    messages = [system_message]

    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)

    # Add current user message
    messages.append(user_message)

    # Get adapter and check supported parameters
    adapter = get_llm_adapter()
    supported_params = adapter.get_supported_params(model)
    response_format = (
        {"type": "json_object"} if "response_format" in supported_params else None
    )

    for attempt in range(max_retries):
        try:
            # Call via adapter
            response = adapter.complete(
                messages=messages,
                model=model,
                api_key=api_key,
                response_format=response_format,
                temperature=0.0,
            )
            
            # Parse the response
            return parse_structured_response(response.content, verbose)

        except Exception as e:
            if verbose:
                print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to get valid structured output after retries: {e}")

    raise ValueError("Max retries exceeded")
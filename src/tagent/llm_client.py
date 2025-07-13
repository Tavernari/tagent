"""
LLM client for TAgent with structured output support via LiteLLM.
"""

from typing import Dict, Optional, Callable, List, Type
from pydantic import BaseModel, ValidationError
import json
import litellm

from .models import StructuredResponse
from .utils import get_tool_documentation

# Enable verbose debug for LLM calls
litellm.log_raw_request_response = False


def query_llm(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    max_retries: int = 3,
    tools: Optional[Dict[str, Callable]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    verbose: bool = False,
) -> StructuredResponse:
    """
    Queries an LLM via LiteLLM and enforces structured output (JSON).
    Checks response_format support dynamically.
    """
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant designed to output JSON. For 'evaluate' actions, include specific feedback. Example: {'action': 'execute', 'params': {'tool': 'tool_name', 'args': {'parameter': 'value'}}, 'reasoning': 'Reason to execute the action.'} or {'action': 'evaluate', 'params': {'achieved': false, 'missing_items': ['item1', 'item2'], 'suggestions': ['suggestion1']}, 'reasoning': 'Detailed explanation of what is missing and why goal is not achieved.'}",
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
            "When using 'execute' action, choose the most appropriate tool based on its documentation. "
            "Ensure 'params' contains 'tool' (tool name) and 'args' (parameters matching the tool's signature).\n"
            "When using 'evaluate' action, if goal is NOT achieved, include 'missing_items' and 'suggestions' in params.\n"
            "Respond ONLY with a valid JSON in the format: "
            "{'action': str (plan|execute|summarize|evaluate), 'params': dict, 'reasoning': str}."
            "Do not add extra text."
        ),
    }

    # Build messages including conversation history
    messages = [system_message]

    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)

    # Add current user message
    messages.append(user_message)

    # Check if model supports response_format
    supported_params = litellm.get_supported_openai_params(model=model)
    response_format = (
        {"type": "json_object"} if "response_format" in supported_params else None
    )

    for attempt in range(max_retries):
        try:
            # Call via LiteLLM, passing api_key if provided
            response = litellm.completion(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=0.0,
                api_key=api_key,
            )
            json_str = response.choices[0].message.content.strip()
            if verbose:
                print(f"[RESPONSE] Raw LLM output: {json_str}")

            # Parse and validate with Pydantic
            return StructuredResponse.model_validate_json(json_str)

        except (
            litellm.AuthenticationError,
            litellm.APIError,
            litellm.ContextWindowExceededError,
            ValidationError,
            json.JSONDecodeError,
        ) as e:
            if verbose:
                print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise ValueError("Failed to get valid structured output after retries")

    raise ValueError("Max retries exceeded")


def query_llm_for_model(
    prompt: str,
    model: str,
    output_model: Type[BaseModel],
    api_key: Optional[str] = None,
    max_retries: int = 3,
    verbose: bool = False,
) -> BaseModel:
    """
    Queries an LLM and enforces the output to conform to a specific Pydantic model.
    """
    # Generate a dummy example based on the schema
    schema = output_model.model_json_schema()
    example_data = {field: f"example_{field}" for field in schema.get("properties", {})}
    example_json = json.dumps(example_data)

    error_feedback = ""
    for attempt in range(max_retries):
        system_message = {
            "role": "system",
            "content": (
                f"You are a helpful assistant designed to output JSON conforming to the following schema: {json.dumps(schema)}.\n"
                f"Example output: {example_json}.\n"
                "Ensure ALL required fields are filled. Do not output empty objects."
            ),
        }

        user_message = {
            "role": "user",
            "content": (
                f"{prompt}\n"
                f"Extract and format data from the state. {error_feedback}\n"
                "Respond ONLY with a valid JSON object matching the schema. No extra text."
            ),
        }

        messages = [system_message, user_message]

        supported_params = litellm.get_supported_openai_params(model=model)
        response_format = (
            {"type": "json_object"} if "response_format" in supported_params else None
        )

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=0.0,
                api_key=api_key,
                model_kwargs=({"strict": True} if "strict" in supported_params else {}),
            )
            json_str = response.choices[0].message.content.strip()
            if verbose:
                print(f"[RESPONSE] Raw LLM output for model query: {json_str}")

            return output_model.model_validate_json(json_str)

        except (
            litellm.AuthenticationError,
            litellm.APIError,
            litellm.ContextWindowExceededError,
            ValidationError,
            json.JSONDecodeError,
        ) as e:
            if verbose:
                print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {e}")
            error_feedback = f"Previous output was invalid: {str(e)}. Correct it by filling all required fields like {list(schema['required'])}."
            if attempt == max_retries - 1:
                raise ValueError("Failed to get valid structured output after retries")

    raise ValueError("Max retries exceeded")


def generate_step_title(
    action: str,
    reasoning: str,
    model: str,
    api_key: Optional[str],
    verbose: bool = False,
) -> str:
    """Generate a concise step title using LLM with token limit for speed/cost."""
    prompt = f"Create a 3-5 word title for this action: {action}. Context: {reasoning[:100]}. Be concise and descriptive."

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise title generator. Respond with ONLY the title, 3-5 words maximum.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
            api_key=api_key,
        )
        title = response.choices[0].message.content.strip()
        return title if title else f"{action.capitalize()} Operation"
    except Exception as e:
        if verbose:
            print(f"[DEBUG] Title generation failed: {e}")
        return f"{action.capitalize()} Operation"

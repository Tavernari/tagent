"""
LLM client for TAgent with structured output support via LiteLLM.
"""

from typing import Dict, Optional, Callable, List, Type
from pydantic import BaseModel, ValidationError
import json
import litellm

from .models import StructuredResponse
from .utils import get_tool_documentation
from .llm_adapter import query_llm_with_adapter

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
    Queries an LLM via adapter pattern and enforces structured output (JSON).
    Uses adapter for easier testing and mocking.
    """
    return query_llm_with_adapter(
        prompt=prompt,
        model=model,
        api_key=api_key,
        max_retries=max_retries,
        tools=tools,
        conversation_history=conversation_history,
        verbose=verbose,
    )


def query_llm_for_model(
    prompt: str,
    model: str,
    output_model: Type[BaseModel],
    api_key: Optional[str] = None,
    max_retries: int = 3,
    verbose: bool = False,
    conversation_history: Optional[List[Dict[str, str]]] = None,  # Added parameter
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
                "Please fill all required fields and provide complete objects."
            ),
        }

        user_message = {
            "role": "user",
            "content": (
                f"{prompt}\n"
                f"Extract and format data from the state. {error_feedback}\n"
                "Please respond with a valid JSON object matching the schema."
            ),
        }

        # Prepend conversation history if available
        messages = conversation_history + [user_message] if conversation_history else [user_message]
        messages.insert(0, system_message)

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
                model_kwargs={"strict": True} if "strict" in supported_params else {},
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
            required_fields = list(schema.get('required', [])) if schema else []
            error_feedback = f"Previous output was invalid: {str(e)}. Correct it by filling all required fields like {required_fields}."
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
    # Extract key information for better titles
    reasoning_short = reasoning[:150] if reasoning else "No specific reasoning"
    
    prompt = (
        f"Create a descriptive 3-6 word title for this {action} step. "
        f"Context: {reasoning_short}. "
        f"Focus on what is being done or accomplished. Examples: "
        f"'Search Flight Prices', 'Compare Retailer Data', 'Generate Strategic Plan'."
    )

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise title generator. Create informative step titles that describe the specific action being taken. Respond with ONLY the title, 3-6 words maximum.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=20,  # Increased for more descriptive titles
            temperature=0.0,
            api_key=api_key,
        )
        title = response.choices[0].message.content.strip()
        # Remove quotes if present
        title = title.strip('"\'')
        return title if title else f"{action.capitalize()} Operation"
    except Exception as e:
        if verbose:
            print(f"[DEBUG] Title generation failed: {e}")
        return f"{action.capitalize()} Operation"


def generate_step_summary(
    action: str,
    reasoning: str,
    result: str,
    model: str,
    api_key: Optional[str],
    verbose: bool = False,
) -> str:
    """
    Generate a concise summary of what happened in this step.
    Uses a low token limit for speed and cost efficiency.
    """
    # Truncate inputs to avoid exceeding context
    reasoning_short = reasoning[:200] if reasoning else "No reasoning provided"
    result_short = result[:300] if result else "No result"
    
    prompt = (
        f"Summarize what was accomplished in this {action} step:\n"
        f"Reasoning: {reasoning_short}\n"
        f"Result: {result_short}\n"
        f"Create a 1-2 sentence summary explaining what was found, decided, or accomplished."
    )

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise step summarizer. Create brief, informative summaries in 1-2 sentences. Focus on key findings and progress.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=60,  # Balance between detail and cost
            temperature=0.0,
            api_key=api_key,
        )
        summary = response.choices[0].message.content.strip()
        return summary if summary else f"Completed {action} step"
    except Exception as e:
        if verbose:
            print(f"[DEBUG] Step summary generation failed: {e}")
        return f"Completed {action} step"

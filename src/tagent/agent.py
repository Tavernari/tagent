# TAgent main module - orchestrates the agent execution loop using task-based approach.
# Integration with LiteLLM for real LLM calls, leveraging JSON Mode.
# Requirements: pip install pydantic litellm
from __future__ import annotations
from typing import Dict, Any, Optional, Callable, Type, Union, TypeVar

from pydantic import BaseModel, Field
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading
    pass

from .version import __version__
from .task_agent import run_task_based_agent, TaskBasedAgentResult, OutputType
from .model_config import AgentModelConfig, create_config_from_string

# Define a TypeVar for the output model, consistent with task_agent
AgentOutputType = TypeVar("AgentOutputType", bound=Optional[BaseModel])

# Export main functions for backwards compatibility
__all__ = ['run_agent', 'TaskBasedAgentResult']


def _parse_model_config(model: Union[str, AgentModelConfig]) -> AgentModelConfig:
    """
    Parse model configuration from string or existing config.
    
    Args:
        model: Model configuration as string or AgentModelConfig object
        
    Returns:
        AgentModelConfig object
    """
    if isinstance(model, str):
        return create_config_from_string(model)
    return model




# === Main Agent Loop ===
def run_agent(
    goal: str,
    config: Optional['TAgentConfig'] = None,
    # Legacy parameters for backward compatibility
    model: Union[str, AgentModelConfig] = "gpt-4",
    api_key: Optional[str] = None,
    max_iterations: int = 20,
    tools: Optional[Dict[str, Callable]] = None,
    output_format: Optional[Type[AgentOutputType]] = None,
    verbose: bool = False,
    crash_if_over_iterations: bool = False,
) -> TaskBasedAgentResult[AgentOutputType]:
    """
    Runs the main agent loop using task-based approach.

    Args:
        goal: The main objective for the agent.
        config: TAgentConfig object containing all configuration options.
                If None, will use legacy parameters and environment variables.
        
        # Legacy parameters (for backward compatibility):
        model: Either a model string (e.g., "gpt-4") for backward compatibility,
            or an AgentModelConfig object for step-specific model configuration.
        api_key: The API key for the LLM service.
        max_iterations: The maximum number of iterations.
        tools: A dictionary of custom tools to register with the agent.
        output_format: The Pydantic model for the final output.
        verbose: If True, shows all debug logs. If False, shows only essential logs.
        crash_if_over_iterations: If True, raises exception when max_iterations
            reached. If False (default), returns results with summarizer fallback.

    Returns:
        TaskBasedAgentResult containing execution results and metadata.
    """
    # Handle configuration: use TAgentConfig if provided, otherwise use legacy parameters
    if config is None:
        # Parse model configuration
        model_config = _parse_model_config(model)
        
        # Use the task-based agent approach
        result = run_task_based_agent(
            goal=goal,
            tools=tools or {},
            output_format=output_format,
            model=model_config.tagent_model,
            api_key=api_key,  # Let LiteLLM handle environment variables
            max_iterations=max_iterations,
            verbose=verbose
        )
        
        return result
    else:
        # Import here to avoid circular imports
        from .config import TAgentConfig
        
        # Override config with any explicitly provided legacy parameters
        override_dict = {}
        if model != "gpt-4":  # Only override if not default
            override_dict["model"] = model
        if api_key is not None:
            override_dict["api_key"] = api_key
        if max_iterations != 20:  # Only override if not default
            override_dict["max_iterations"] = max_iterations
        if tools is not None:
            override_dict["tools"] = tools
        if output_format is not None:
            override_dict["output_format"] = output_format
        if verbose:  # Only override if True
            override_dict["verbose"] = verbose
        if crash_if_over_iterations:  # Only override if True
            override_dict["crash_if_over_iterations"] = crash_if_over_iterations
        
        if override_dict:
            config = config.merge(TAgentConfig.from_dict(override_dict))
        
        # Extract values from config
        model_config = config.get_model_config()
        max_iterations = config.max_iterations
        tools = config.tools
        output_format = config.output_format
        verbose = config.verbose
        
        # Set UI style
        from .ui import set_ui_style
        set_ui_style(config.ui_style)
        
        # Use the task-based agent approach
        result = run_task_based_agent(
            goal=goal,
            tools=tools or {},
            output_format=output_format,
            model=model_config.tagent_model,
            api_key=model_config.api_key,
            max_iterations=max_iterations,
            verbose=verbose
        )
        
        return result
    


# === Example Usage ===
if __name__ == "__main__":
    import time

    # Define a fake tool to fetch weather data with a delay
    def fetch_weather_tool(
        state: Dict[str, Any], args: Dict[str, Any]
    ) -> Optional[tuple]:
        location = args.get("location", "default")
        print(f"[INFO] Fetching weather for {location}...")
        time.sleep(3)
        # Simulated weather data
        weather_data = {
            "location": location,
            "temperature": "25°C",
            "condition": "Sunny",
        }
        print(f"[INFO] Weather data fetched for {location}.")
        # Note: state parameter is available for accessing agent state if needed
        return ("weather_data", weather_data)

    # Create a dictionary of tools to register
    agent_tools = {"fetch_weather": fetch_weather_tool}

    # Define the desired output format
    class WeatherReport(BaseModel):
        location: str = Field(..., description="The location of the weather report.")
        temperature: str = Field(..., description="The temperature in Celsius.")
        condition: str = Field(..., description="The weather condition.")
        summary: str = Field(..., description="A summary of the weather report.")

    # Create the agent and pass the tools and output format
    agent_goal = "Create a weather report for London."
    result = run_agent(
        goal=agent_goal,
        model="gpt-4",
        tools=agent_tools,
        output_format=WeatherReport,
        verbose=True
    )
    print("\nFinal Result:", result)
    if result.output:
        print(f"Location: {result.output.location}")

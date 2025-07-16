# TAgent main module - orchestrates the agent execution loop using task-based approach.
# Integration with LiteLLM for real LLM calls, leveraging JSON Mode.
# Requirements: pip install pydantic litellm
from __future__ import annotations
from typing import Dict, Any, Optional, Callable, Type, Union, TypeVar
import asyncio
import logging

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

# Graceful import handling for pipeline functionality
try:
    from .pipeline import (
        Pipeline, PipelineResult, PipelineExecutor, 
        PipelineExecutorConfig, PipelineValidationError
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    
    # Create dummy classes for type hints when pipeline is not available
    class Pipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Pipeline support requires additional dependencies. "
                "Please ensure all pipeline modules are properly installed."
            )
    
    class PipelineResult:
        pass

# Define a TypeVar for the output model, consistent with task_agent
AgentOutputType = TypeVar("AgentOutputType", bound=Optional[BaseModel])

# Export main functions for backwards compatibility
__all__ = ['run_agent', 'run_pipeline', 'TaskBasedAgentResult', 'PipelineResult', 'Pipeline']

logger = logging.getLogger(__name__)


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
    goal_or_pipeline: Union[str, Pipeline],
    config: Optional['TAgentConfig'] = None,
    # Legacy parameters for backward compatibility
    model: Union[str, AgentModelConfig] = "gpt-4",
    api_key: Optional[str] = None,
    max_iterations: int = 20,
    tools: Optional[Dict[str, Callable]] = None,
    output_format: Optional[Type[AgentOutputType]] = None,
    verbose: bool = False,
    crash_if_over_iterations: bool = False,
    # Pipeline-specific parameters
    executor_config: Optional['PipelineExecutorConfig'] = None,
) -> Union[TaskBasedAgentResult[AgentOutputType], PipelineResult]:
    """
    Enhanced run_agent supporting both single goals and pipelines.

    Args:
        goal_or_pipeline: Either a goal string or Pipeline object
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
        
        # Pipeline-specific parameters:
        executor_config: Optional PipelineExecutorConfig for pipeline execution

    Returns:
        TaskBasedAgentResult for single goals or PipelineResult for pipelines
        
    Raises:
        ImportError: If pipeline functionality is not available
        PipelineValidationError: If pipeline validation fails
    """
    # Check if pipeline functionality is available and requested
    if isinstance(goal_or_pipeline, Pipeline):
        if not PIPELINE_AVAILABLE:
            raise ImportError(
                "Pipeline support requires additional dependencies. "
                "Please ensure all pipeline modules are properly installed."
            )
        
        # Execute pipeline
        return asyncio.run(run_pipeline(
            pipeline=goal_or_pipeline,
            config=config,
            model=model,
            api_key=api_key,
            max_iterations=max_iterations,
            tools=tools,
            output_format=output_format,
            verbose=verbose,
            crash_if_over_iterations=crash_if_over_iterations,
            executor_config=executor_config
        ))
    else:
        # Existing single-goal execution
        goal = goal_or_pipeline
        
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
            
            # Create override config using TAgentConfig for type safety
            override_config = TAgentConfig()
            if model != "gpt-4":  # Only override if not default
                override_config.model = model
            if api_key is not None:
                override_config.api_key = api_key
            if max_iterations != 20:  # Only override if not default
                override_config.max_iterations = max_iterations
            if tools is not None:
                override_config.tools = tools
            if output_format is not None:
                override_config.output_format = output_format
            if verbose:  # Only override if True
                override_config.verbose = verbose
            if crash_if_over_iterations:  # Only override if True
                override_config.crash_if_over_iterations = crash_if_over_iterations
            
            # Merge configurations
            merged_config = config.merge(override_config)
            
            # Extract values from config
            model_config = merged_config.get_model_config()
            
            # Set UI style
            from .ui import set_ui_style
            set_ui_style(merged_config.ui_style)
            
            # Use the task-based agent approach
            result = run_task_based_agent(
                goal=goal,
                tools=merged_config.tools or {},
                output_format=merged_config.output_format,
                model=model_config.tagent_model,
                api_key=model_config.api_key,
                max_iterations=merged_config.max_iterations,
                verbose=merged_config.verbose
            )
            
            return result


async def run_pipeline(
    pipeline: Pipeline,
    config: Optional['TAgentConfig'] = None,
    # Legacy parameters for backward compatibility  
    model: Union[str, AgentModelConfig] = "gpt-4",
    api_key: Optional[str] = None,
    max_iterations: int = 20,
    tools: Optional[Dict[str, Callable]] = None,
    output_format: Optional[Type[AgentOutputType]] = None,
    verbose: bool = False,
    crash_if_over_iterations: bool = False,
    executor_config: Optional['PipelineExecutorConfig'] = None,
) -> PipelineResult:
    """
    Execute a pipeline with enhanced orchestration.

    Args:
        pipeline: Pipeline object to execute
        config: TAgentConfig object containing all configuration options
        model: Model configuration for pipeline execution
        api_key: API key for LLM service
        max_iterations: Maximum iterations per step
        tools: Dictionary of available tools
        output_format: Output format for pipeline results
        verbose: Enable verbose logging
        crash_if_over_iterations: Crash if iterations exceeded
        executor_config: Configuration for pipeline executor

    Returns:
        PipelineResult with execution results and metadata
        
    Raises:
        PipelineValidationError: If pipeline validation fails
        ImportError: If pipeline dependencies are not available
    """
    if not PIPELINE_AVAILABLE:
        raise ImportError(
            "Pipeline support requires additional dependencies. "
            "Please ensure all pipeline modules are properly installed."
        )
    
    # Validate pipeline before execution
    validation_errors = pipeline.validate()
    if validation_errors:
        raise PipelineValidationError(f"Pipeline validation failed: {validation_errors}")
    
    # Prepare configuration with type safety
    if config is None:
        # Import here to avoid circular imports
        from .config import TAgentConfig
        
        # Create config from legacy parameters
        config = TAgentConfig(
            model=model,
            api_key=api_key,
            max_iterations=max_iterations,
            tools=tools,
            output_format=output_format,
            verbose=verbose,
            crash_if_over_iterations=crash_if_over_iterations
        )
    else:
        # Import here to avoid circular imports
        from .config import TAgentConfig
        
        # Create override config for type safety
        override_config = TAgentConfig()
        if model != "gpt-4":
            override_config.model = model
        if api_key is not None:
            override_config.api_key = api_key
        if max_iterations != 20:
            override_config.max_iterations = max_iterations
        if tools is not None:
            override_config.tools = tools
        if output_format is not None:
            override_config.output_format = output_format
        if verbose:
            override_config.verbose = verbose
        if crash_if_over_iterations:
            override_config.crash_if_over_iterations = crash_if_over_iterations
        
        # Merge configurations
        config = config.merge(override_config)
    
    # Set UI style
    from .ui import set_ui_style
    set_ui_style(config.ui_style)
    
    # Create executor configuration if not provided
    if executor_config is None:
        executor_config = PipelineExecutorConfig()
    
    # Create and execute pipeline
    executor = PipelineExecutor(pipeline, config, executor_config)
    
    try:
        result = await executor.execute()
        logger.info(f"Pipeline '{pipeline.name}' executed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Pipeline '{pipeline.name}' execution failed: {e}")
        raise


def _merge_config_with_kwargs(config: 'TAgentConfig', **kwargs) -> 'TAgentConfig':
    """
    Merge configuration with kwargs for backward compatibility using type-safe approach.
    
    Args:
        config: Base TAgentConfig object
        **kwargs: Additional parameters to merge
        
    Returns:
        Merged TAgentConfig object
    """
    # Import here to avoid circular imports
    from .config import TAgentConfig
    
    # Create override config
    override_config = TAgentConfig()
    
    # Map kwargs to config parameters with type safety
    if 'model' in kwargs:
        override_config.model = kwargs['model']
    if 'api_key' in kwargs:
        override_config.api_key = kwargs['api_key']
    if 'max_iterations' in kwargs:
        override_config.max_iterations = kwargs['max_iterations']
    if 'tools' in kwargs:
        override_config.tools = kwargs['tools']
    if 'output_format' in kwargs:
        override_config.output_format = kwargs['output_format']
    if 'verbose' in kwargs:
        override_config.verbose = kwargs['verbose']
    if 'crash_if_over_iterations' in kwargs:
        override_config.crash_if_over_iterations = kwargs['crash_if_over_iterations']
    
    # Merge configurations
    return config.merge(override_config)



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

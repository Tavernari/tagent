"""
Model configuration system with step-specific overrides.

This module provides a centralized way to configure models for different 
agent steps (planner, executor, summarizer, evaluator, finalizer) with
fallback to a global model configuration.

Environment Variables:
    TAGENT_MODEL: Global fallback model for all steps
    TAGENT_PLANNER_MODEL: Model for planning actions
    TAGENT_EXECUTOR_MODEL: Model for execution actions  
    TAGENT_SUMMARIZER_MODEL: Model for summarization actions
    TAGENT_EVALUATOR_MODEL: Model for evaluation actions
    TAGENT_FINALIZER_MODEL: Model for final formatting actions
"""

import os
from typing import Optional, Dict, Union
from enum import Enum
from pydantic import BaseModel


class AgentStep(Enum):
    """Agent step types for model configuration."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    SUMMARIZER = "summarizer"
    EVALUATOR = "evaluator"
    FINALIZER = "finalizer"


class AgentModelConfig(BaseModel):
    """
    Pydantic model for agent configuration with step-specific model overrides.
    
    This configuration object allows specifying different models for each agent step,
    with a required global model as fallback. When passed to run_agent, these models
    take precedence over environment variables.
    
    Attributes:
        tagent_model: Required global model for all steps (fallback)
        tagent_planner_model: Optional model for planning actions
        tagent_executor_model: Optional model for execution actions
        tagent_summarizer_model: Optional model for summarization actions  
        tagent_evaluator_model: Optional model for evaluation actions
        tagent_finalizer_model: Optional model for final formatting actions
        api_key: Optional API key for the LLM service
        
    Example:
        >>> config = AgentModelConfig(
        ...     tagent_model="gpt-4",
        ...     tagent_planner_model="gpt-4-turbo",
        ...     tagent_evaluator_model="gpt-4",
        ...     api_key="sk-..."
        ... )
        >>> # Other steps will use gpt-4, planner uses gpt-4-turbo
    """
    
    tagent_model: str  # Required global fallback model
    tagent_planner_model: Optional[str] = None
    tagent_executor_model: Optional[str] = None
    tagent_summarizer_model: Optional[str] = None
    tagent_evaluator_model: Optional[str] = None
    tagent_finalizer_model: Optional[str] = None
    api_key: Optional[str] = None


class ModelConfig:
    """Centralized model configuration with step-specific overrides."""
    
    # Default fallback model if no environment variables are set
    DEFAULT_MODEL = "gpt-3.5-turbo"
    
    # Environment variable names
    GLOBAL_MODEL_ENV = "TAGENT_MODEL"
    STEP_MODEL_ENVS = {
        AgentStep.PLANNER: "TAGENT_PLANNER_MODEL",
        AgentStep.EXECUTOR: "TAGENT_EXECUTOR_MODEL", 
        AgentStep.SUMMARIZER: "TAGENT_SUMMARIZER_MODEL",
        AgentStep.EVALUATOR: "TAGENT_EVALUATOR_MODEL",
        AgentStep.FINALIZER: "TAGENT_FINALIZER_MODEL",
    }
    
    @classmethod
    def get_model_for_step(
        cls, 
        step: AgentStep, 
        fallback_model: Optional[str] = None,
        config: Optional[AgentModelConfig] = None
    ) -> str:
        """
        Get the appropriate model for a specific agent step.
        
        Priority order:
        1. Step-specific model from config object (highest priority)
        2. Global model from config object 
        3. Step-specific environment variable (e.g., TAGENT_PLANNER_MODEL)
        4. Global environment variable (TAGENT_MODEL)
        5. Provided fallback_model parameter
        6. Default model
        
        Args:
            step: The agent step requiring a model
            fallback_model: Optional fallback model if config/env vars not set
            config: Optional configuration object with model overrides
            
        Returns:
            Model name string to use for the specified step
            
        Example:
            >>> config = AgentModelConfig(tagent_model='gpt-4', tagent_planner_model='gpt-4-turbo')
            >>> ModelConfig.get_model_for_step(AgentStep.PLANNER, config=config)
            'gpt-4-turbo'
            >>> ModelConfig.get_model_for_step(AgentStep.EXECUTOR, config=config)
            'gpt-4'
        """
        # Priority 1: Step-specific model from config object
        if config:
            step_attr_map = {
                AgentStep.PLANNER: config.tagent_planner_model,
                AgentStep.EXECUTOR: config.tagent_executor_model,
                AgentStep.SUMMARIZER: config.tagent_summarizer_model,
                AgentStep.EVALUATOR: config.tagent_evaluator_model,
                AgentStep.FINALIZER: config.tagent_finalizer_model,
            }
            step_model = step_attr_map.get(step)
            if step_model:
                return step_model
                
            # Priority 2: Global model from config object
            if config.tagent_model:
                return config.tagent_model
        
        # Priority 3: Step-specific environment variable
        step_env = cls.STEP_MODEL_ENVS.get(step)
        if step_env:
            step_model = os.getenv(step_env)
            if step_model:
                return step_model
        
        # Priority 4: Global environment variable
        global_model = os.getenv(cls.GLOBAL_MODEL_ENV)
        if global_model:
            return global_model
            
        # Priority 5: Provided fallback
        if fallback_model:
            return fallback_model
            
        # Priority 6: Default
        return cls.DEFAULT_MODEL
    
    @classmethod
    def get_all_configured_models(cls) -> Dict[str, str]:
        """
        Get all currently configured models from environment variables.
        
        Returns:
            Dictionary mapping step names to configured models
        """
        config = {}
        
        # Global model
        global_model = os.getenv(cls.GLOBAL_MODEL_ENV)
        if global_model:
            config["global"] = global_model
            
        # Step-specific models
        for step, env_var in cls.STEP_MODEL_ENVS.items():
            step_model = os.getenv(env_var)
            if step_model:
                config[step.value] = step_model
                
        return config
    
    @classmethod 
    def get_effective_models(cls, fallback_model: Optional[str] = None) -> Dict[str, str]:
        """
        Get the effective model that would be used for each step.
        
        Args:
            fallback_model: Optional fallback model if env vars not set
            
        Returns:
            Dictionary mapping step names to effective models
        """
        return {
            step.value: cls.get_model_for_step(step, fallback_model)
            for step in AgentStep
        }


def get_planner_model(
    fallback_model: Optional[str] = None, 
    config: Optional[AgentModelConfig] = None
) -> str:
    """Get model for planning actions."""
    return ModelConfig.get_model_for_step(AgentStep.PLANNER, fallback_model, config)


def get_executor_model(
    fallback_model: Optional[str] = None,
    config: Optional[AgentModelConfig] = None
) -> str:
    """Get model for execution actions."""
    return ModelConfig.get_model_for_step(AgentStep.EXECUTOR, fallback_model, config)


def get_summarizer_model(
    fallback_model: Optional[str] = None,
    config: Optional[AgentModelConfig] = None
) -> str:
    """Get model for summarization actions."""
    return ModelConfig.get_model_for_step(AgentStep.SUMMARIZER, fallback_model, config)


def get_evaluator_model(
    fallback_model: Optional[str] = None,
    config: Optional[AgentModelConfig] = None
) -> str:
    """Get model for evaluation actions."""
    return ModelConfig.get_model_for_step(AgentStep.EVALUATOR, fallback_model, config)


def get_finalizer_model(
    fallback_model: Optional[str] = None,
    config: Optional[AgentModelConfig] = None
) -> str:
    """Get model for final formatting actions."""
    return ModelConfig.get_model_for_step(AgentStep.FINALIZER, fallback_model, config)


def create_config_from_string(
    model: str, 
    api_key: Optional[str] = None
) -> AgentModelConfig:
    """
    Create an AgentModelConfig from a simple model string.
    
    This is used when the user passes a string instead of a config object
    to run_agent, maintaining backward compatibility.
    
    Args:
        model: The model name to use for all steps
        api_key: Optional API key
        
    Returns:
        AgentModelConfig object with the model set as global fallback
    """
    return AgentModelConfig(tagent_model=model, api_key=api_key)
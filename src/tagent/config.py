"""
TAgent Configuration System

This module provides a centralized configuration system for TAgent,
allowing users to configure various aspects of the agent behavior
including UI style, model settings, and execution parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable, Type
from pydantic import BaseModel

from .ui.factory import UIStyle
from .model_config import AgentModelConfig


@dataclass
class TAgentConfig:
    """
    Comprehensive configuration for TAgent.
    
    This class centralizes all configuration options for TAgent,
    providing a single source of truth for agent behavior settings.
    """
    
    # Core execution parameters
    max_iterations: int = 20
    verbose: bool = False
    crash_if_over_iterations: bool = False
    
    # Model configuration
    model: Optional[Union[str, AgentModelConfig]] = None
    api_key: Optional[str] = None
    
    # UI configuration
    ui_style: UIStyle = UIStyle.ANIMATED
    
    # Tool configuration
    tools: Optional[Dict[str, Callable]] = None
    
    # Output configuration
    output_format: Optional[Type[BaseModel]] = None
    
    # Advanced settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> 'TAgentConfig':
        """
        Create configuration from environment variables.
        
        Environment variables:
        - TAGENT_MAX_ITERATIONS: Maximum number of iterations
        - TAGENT_VERBOSE: Enable verbose output (true/false)
        - TAGENT_CRASH_IF_OVER_ITERATIONS: Crash if over iterations (true/false)
        - TAGENT_MODEL: Default model to use
        - TAGENT_API_KEY: API key for model access
        - TAGENT_UI_STYLE: UI style (animated/institutional)
        
        Returns:
            TAgentConfig instance with values from environment variables
        """
        config = cls()
        
        # Core execution parameters
        if max_iter := os.getenv("TAGENT_MAX_ITERATIONS"):
            config.max_iterations = int(max_iter)
        
        if verbose := os.getenv("TAGENT_VERBOSE"):
            config.verbose = verbose.lower() in ("true", "1", "yes", "on")
        
        if crash_over := os.getenv("TAGENT_CRASH_IF_OVER_ITERATIONS"):
            config.crash_if_over_iterations = crash_over.lower() in ("true", "1", "yes", "on")
        
        # Model configuration
        if model := os.getenv("TAGENT_MODEL"):
            config.model = model
        
        if api_key := os.getenv("TAGENT_API_KEY"):
            config.api_key = api_key
        
        # UI configuration
        if ui_style := os.getenv("TAGENT_UI_STYLE"):
            ui_style_map = {
                "animated": UIStyle.ANIMATED,
                "institutional": UIStyle.INSTITUTIONAL,
                "matrix": UIStyle.ANIMATED,
                "server": UIStyle.INSTITUTIONAL,
                "log": UIStyle.INSTITUTIONAL,
                "logging": UIStyle.INSTITUTIONAL,
            }
            config.ui_style = ui_style_map.get(ui_style.lower(), UIStyle.ANIMATED)
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TAgentConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            TAgentConfig instance with values from dictionary
        """
        config = cls()
        
        # Update fields that exist in the dataclass
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Store unknown keys in custom_settings
                config.custom_settings[key] = value
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        result = {}
        
        # Add all dataclass fields
        for field_info in self.__dataclass_fields__.values():
            value = getattr(self, field_info.name)
            if field_info.name == "ui_style" and isinstance(value, UIStyle):
                result[field_info.name] = value.value
            else:
                result[field_info.name] = value
        
        # Add custom settings
        result.update(self.custom_settings)
        
        return result
    
    def merge(self, other: 'TAgentConfig') -> 'TAgentConfig':
        """
        Merge this configuration with another, with other taking precedence.
        
        Args:
            other: Configuration to merge with this one
            
        Returns:
            New TAgentConfig instance with merged values
        """
        config = TAgentConfig()
        
        # Copy values from self
        for field_info in self.__dataclass_fields__.values():
            setattr(config, field_info.name, getattr(self, field_info.name))
        
        # Override with values from other
        for field_info in other.__dataclass_fields__.values():
            other_value = getattr(other, field_info.name)
            if field_info.name == "custom_settings":
                # Merge custom settings
                config.custom_settings.update(other_value)
            elif field_info.name == "tools" and other_value is not None:
                # Merge tools
                if config.tools is None:
                    config.tools = {}
                config.tools.update(other_value)
            elif other_value != field_info.default:
                # Override if not default value
                setattr(config, field_info.name, other_value)
        
        return config
    
    def get_model_config(self) -> AgentModelConfig:
        """
        Get the model configuration for this TAgent config.
        
        Returns:
            AgentModelConfig instance based on the model setting
        """
        if isinstance(self.model, str):
            # Import here to avoid circular imports
            from .model_config import create_config_from_string
            return create_config_from_string(self.model, self.api_key)
        else:
            return self.model
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure UI style is UIStyle enum
        if isinstance(self.ui_style, str):
            ui_style_map = {
                "animated": UIStyle.ANIMATED,
                "institutional": UIStyle.INSTITUTIONAL,
                "matrix": UIStyle.ANIMATED,
                "server": UIStyle.INSTITUTIONAL,
                "log": UIStyle.INSTITUTIONAL,
                "logging": UIStyle.INSTITUTIONAL,
            }
            self.ui_style = ui_style_map.get(self.ui_style.lower(), UIStyle.ANIMATED)


def load_config(
    config_path: Optional[str] = None,
    env_override: bool = True,
    **kwargs
) -> TAgentConfig:
    """
    Load configuration from various sources.
    
    Args:
        config_path: Path to configuration file (JSON/YAML)
        env_override: Whether environment variables should override config file
        **kwargs: Additional configuration overrides
        
    Returns:
        TAgentConfig instance with loaded configuration
        
    Loading priority (higher overrides lower):
    1. Config file (if provided)
    2. Environment variables (if env_override=True)
    3. Keyword arguments
    """
    config = TAgentConfig()
    
    # Load from file if provided
    if config_path:
        import json
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    file_config = json.load(f)
                else:
                    # Try to import yaml for YAML support
                    try:
                        import yaml
                        file_config = yaml.safe_load(f)
                    except ImportError:
                        raise ImportError("PyYAML is required for YAML config files")
                
                config = config.merge(TAgentConfig.from_dict(file_config))
        except FileNotFoundError:
            pass  # Config file is optional
    
    # Override with environment variables
    if env_override:
        config = config.merge(TAgentConfig.from_env())
    
    # Override with keyword arguments
    if kwargs:
        config = config.merge(TAgentConfig.from_dict(kwargs))
    
    return config
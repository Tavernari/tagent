import os
from enum import Enum
from typing import Optional

from .interface import UIInterface
from .animated import AnimatedUI
from .institutional import InstitutionalUI


class UIStyle(Enum):
    """Available UI styles for TAgent."""
    ANIMATED = "animated"
    INSTITUTIONAL = "institutional"
    MATRIX = "animated"  # Alias for animated


class UIFactory:
    """Factory for creating UI instances based on style preference."""
    
    @staticmethod
    def create_ui(style: Optional[UIStyle] = None) -> UIInterface:
        """
        Create a UI instance based on the specified style.
        
        Args:
            style: The UI style to use. If None, will try to detect from environment
                  variables or default to ANIMATED.
        
        Returns:
            A UI instance implementing the UIInterface.
        
        Environment Variables:
            TAGENT_UI_STYLE: Can be set to "animated", "institutional", or "matrix"
        """
        if style is None:
            style = UIFactory._detect_style_from_env()
        
        if style == UIStyle.INSTITUTIONAL:
            return InstitutionalUI()
        else:
            return AnimatedUI()
    
    @staticmethod
    def _detect_style_from_env() -> UIStyle:
        """Detect UI style from environment variables."""
        env_style = os.getenv("TAGENT_UI_STYLE", "animated").lower()
        
        style_map = {
            "animated": UIStyle.ANIMATED,
            "institutional": UIStyle.INSTITUTIONAL,
            "matrix": UIStyle.MATRIX,
            "server": UIStyle.INSTITUTIONAL,
            "log": UIStyle.INSTITUTIONAL,
            "logging": UIStyle.INSTITUTIONAL,
        }
        
        return style_map.get(env_style, UIStyle.ANIMATED)
    
    @staticmethod
    def get_available_styles() -> list[str]:
        """Get a list of available UI styles."""
        return [style.value for style in UIStyle if style != UIStyle.MATRIX]  # Exclude alias
    
    @staticmethod
    def create_animated_ui() -> UIInterface:
        """Create an animated (Matrix-style) UI instance."""
        return AnimatedUI()
    
    @staticmethod
    def create_institutional_ui() -> UIInterface:
        """Create an institutional (server-focused) UI instance."""
        return InstitutionalUI()
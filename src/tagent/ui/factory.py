import os
from enum import Enum
from typing import Optional

from .interface import UIInterface
from .modern_cli import ModernCliUI
from .institutional import InstitutionalUI


class UIStyle(Enum):
    """Available UI styles for TAgent."""
    MODERN = "modern"
    INSTITUTIONAL = "institutional"


class UIFactory:
    """Factory for creating UI instances based on style preference."""
    
    @staticmethod
    def create_ui(style: Optional[UIStyle] = None) -> UIInterface:
        """
        Create a UI instance based on the specified style.
        
        Args:
            style: The UI style to use. If None, will try to detect from environment
                  variables or default to MODERN.
        
        Returns:
            A UI instance implementing the UIInterface.
        
        Environment Variables:
            TAGENT_UI_STYLE: Can be set to "modern" or "institutional"
        """
        if style is None:
            style = UIFactory._detect_style_from_env()
        
        if style == UIStyle.INSTITUTIONAL:
            return InstitutionalUI()
        else:
            return ModernCliUI()
    
    @staticmethod
    def _detect_style_from_env() -> UIStyle:
        """Detect UI style from environment variables."""
        env_style = os.getenv("TAGENT_UI_STYLE", "modern").lower()
        
        style_map = {
            "modern": UIStyle.MODERN,
            "cli": UIStyle.MODERN,
            "vscode": UIStyle.MODERN,
            "animated": UIStyle.MODERN,  # Legacy support
            "matrix": UIStyle.MODERN,  # Legacy support
            "institutional": UIStyle.INSTITUTIONAL,
            "server": UIStyle.INSTITUTIONAL,
            "log": UIStyle.INSTITUTIONAL,
        }
        
        return style_map.get(env_style, UIStyle.MODERN)
    
    @staticmethod
    def get_available_styles() -> list[str]:
        """Get a list of available UI styles."""
        return [style.value for style in UIStyle]
    
    @staticmethod
    def create_modern_ui() -> UIInterface:
        """Create a modern CLI UI instance."""
        return ModernCliUI()
    
    @staticmethod
    def create_institutional_ui() -> UIInterface:
        """Create an institutional (server-focused) UI instance."""
        return InstitutionalUI()

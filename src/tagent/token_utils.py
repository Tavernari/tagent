"""
Token utilities for TAgent using litellm for accurate token counting.

This module provides token-based metrics to replace character-based metrics
for better LLM compatibility and more accurate context management.
"""

from typing import Optional, Union
from litellm import get_max_tokens, token_counter
import logging

logger = logging.getLogger(__name__)


def get_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Get the token count for a given text using litellm.
    
    Args:
        text: The text to count tokens for
        model: The model to use for token counting (default: gpt-3.5-turbo)
        
    Returns:
        Number of tokens in the text
    """
    try:
        return token_counter(model=model, text=text)
    except Exception as e:
        logger.warning(f"Failed to count tokens using litellm: {e}")
        # Fallback to rough estimate: ~4 characters per token
        return len(text) // 4


def get_model_max_tokens(model: str = "gpt-3.5-turbo") -> int:
    """
    Get the maximum token limit for a given model.
    
    Args:
        model: The model to get the token limit for
        
    Returns:
        Maximum number of tokens the model can handle
    """
    try:
        return get_max_tokens(model)
    except Exception as e:
        logger.warning(f"Failed to get max tokens for model {model}: {e}")
        # Common fallback limits
        fallback_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "claude-3-haiku": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-opus": 200000,
        }
        return fallback_limits.get(model, 4096)


def truncate_text_by_tokens(
    text: str, 
    max_tokens: int, 
    model: str = "gpt-3.5-turbo",
    suffix: str = "..."
) -> str:
    """
    Truncate text to fit within a token limit.
    
    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed
        model: The model to use for token counting
        suffix: Suffix to add when text is truncated
        
    Returns:
        Truncated text that fits within the token limit
    """
    current_tokens = get_token_count(text, model)
    
    if current_tokens <= max_tokens:
        return text
    
    # Binary search to find the right truncation point
    left, right = 0, len(text)
    result = text
    
    while left < right:
        mid = (left + right + 1) // 2
        candidate = text[:mid] + suffix
        candidate_tokens = get_token_count(candidate, model)
        
        if candidate_tokens <= max_tokens:
            result = candidate
            left = mid
        else:
            right = mid - 1
    
    return result


def format_token_size_info(text: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Format token size information for logging/display.
    
    Args:
        text: The text to analyze
        model: The model to use for token counting
        
    Returns:
        Formatted string with token count and character count
    """
    token_count = get_token_count(text, model)
    char_count = len(text)
    return f"{token_count} tokens ({char_count} chars)"


def is_within_token_limit(
    text: str, 
    max_tokens: int, 
    model: str = "gpt-3.5-turbo"
) -> bool:
    """
    Check if text is within the specified token limit.
    
    Args:
        text: The text to check
        max_tokens: Maximum number of tokens allowed
        model: The model to use for token counting
        
    Returns:
        True if text is within the limit, False otherwise
    """
    return get_token_count(text, model) <= max_tokens


def get_optimal_context_size(
    texts: list[str], 
    max_total_tokens: int, 
    model: str = "gpt-3.5-turbo"
) -> list[str]:
    """
    Select texts that fit within a total token budget, prioritizing by order.
    
    Args:
        texts: List of texts to consider
        max_total_tokens: Maximum total tokens for all texts combined
        model: The model to use for token counting
        
    Returns:
        List of texts that fit within the token budget
    """
    selected_texts = []
    total_tokens = 0
    
    for text in texts:
        text_tokens = get_token_count(text, model)
        if total_tokens + text_tokens <= max_total_tokens:
            selected_texts.append(text)
            total_tokens += text_tokens
        else:
            break
    
    return selected_texts
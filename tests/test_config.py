"""
Configuration for end-to-end tests.
"""

import os
from typing import Optional

def get_openrouter_api_key() -> Optional[str]:
    """
    Get OpenRouter API key from environment variable.
    
    To run tests with real OpenRouter:
    1. Get API key from https://openrouter.ai/
    2. Set environment variable: export OPENROUTER_API_KEY="your-key-here"
    
    Returns:
        API key if available, None otherwise
    """
    return os.getenv("OPENROUTER_API_KEY")

def should_run_openrouter_tests() -> bool:
    """
    Check if OpenRouter tests should be run.
    """
    return get_openrouter_api_key() is not None

# Test configuration
OPENROUTER_MODEL = "openrouter/qwen/qwen3-8b"
MAX_ITERATIONS = 7
PRODUCT_NAME = "Wireless Headphones"

# Expected results for validation
EXPECTED_RETAIL_A_PRICE = 10.0  # USD
EXPECTED_RETAIL_B_PRICE_EUR = 6.0  # EUR
EXPECTED_RETAIL_B_PRICE_USD = 6.6  # ~6.6 USD after conversion
EXPECTED_CHEAPEST = "Retail B"
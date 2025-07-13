"""
End-to-end test for TAgent price comparison across retailers with currency conversion.

This test validates:
1. Agent can reason about price comparison task
2. Agent plans to gather prices from multiple retailers  
3. Agent acts by calling retail price tools and currency conversion
4. Agent completes task within max iterations (7)
5. Agent produces correct final answer
6. Tools are called the expected number of times
"""

import pytest
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from src.tagent.agent import run_agent
from test_config import get_openrouter_api_key


# === Mock Tools with Spy Functionality ===

class ToolSpy:
    """Spy class to track tool invocations."""
    
    def __init__(self):
        self.call_count = 0
        self.calls = []
        
    def reset(self):
        self.call_count = 0
        self.calls = []
        
    def record_call(self, tool_name: str, args: Dict[str, Any]):
        self.call_count += 1
        self.calls.append({"tool": tool_name, "args": args, "call_number": self.call_count})


# Global spies for tracking tool calls
retail_a_spy = ToolSpy()
retail_b_spy = ToolSpy()
currency_converter_spy = ToolSpy()


def get_retail_a_price_tool(_state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Mock tool that returns price from Retail A in USD.
    
    Args:
        state: Current agent state
        args: Tool arguments containing 'product' name
        
    Returns:
        Tuple with price data from Retail A
    """
    retail_a_spy.record_call("get_retail_a_price", args)
    
    product = args.get("product", "unknown_product")
    
    # Mock price data - Retail A is more expensive (10 USD)
    price_data = {
        "retailer": "Retail A",
        "product": product,
        "price": 10.0,
        "currency": "USD",
        "availability": "In Stock"
    }
    
    return ("retail_a_price", price_data)


def get_retail_b_price_tool(_state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Mock tool that returns price from Retail B in EUR.
    
    Args:
        state: Current agent state
        args: Tool arguments containing 'product' name
        
    Returns:
        Tuple with price data from Retail B
    """
    retail_b_spy.record_call("get_retail_b_price", args)
    
    product = args.get("product", "unknown_product")
    
    # Mock price data - Retail B is cheaper (6 EUR, which should be ~6.60 USD)
    price_data = {
        "retailer": "Retail B", 
        "product": product,
        "price": 6.0,
        "currency": "EUR",
        "availability": "In Stock"
    }
    
    return ("retail_b_price", price_data)


def convert_currency_tool(_state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Mock tool that converts between currencies.
    
    Args:
        state: Current agent state
        args: Tool arguments with 'amount', 'from_currency', 'to_currency'
        
    Returns:
        Tuple with conversion result
    """
    currency_converter_spy.record_call("convert_currency", args)
    
    amount = args.get("amount", 0)
    from_currency = args.get("from_currency", "USD")
    to_currency = args.get("to_currency", "USD")
    
    # Mock exchange rates
    exchange_rates = {
        ("EUR", "USD"): 1.10,  # 1 EUR = 1.10 USD
        ("USD", "EUR"): 0.91,  # 1 USD = 0.91 EUR
        ("USD", "USD"): 1.0,   # Same currency
        ("EUR", "EUR"): 1.0,   # Same currency
    }
    
    rate = exchange_rates.get((from_currency, to_currency), 1.0)
    converted_amount = amount * rate
    
    conversion_data = {
        "original_amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "exchange_rate": rate,
        "converted_amount": converted_amount
    }
    
    return ("currency_conversion", conversion_data)


# === Output Model ===

class PriceComparisonResult(BaseModel):
    """Expected output format for price comparison task."""
    
    product_name: str = Field(..., description="Name of the product being compared")
    retail_a_price_usd: float = Field(..., description="Price from Retail A in USD")
    retail_b_price_usd: float = Field(..., description="Price from Retail B converted to USD")
    cheapest_retailer: str = Field(..., description="Name of the retailer with the lowest price")
    price_difference_usd: float = Field(..., description="Price difference in USD between retailers")
    summary: str = Field(..., description="Brief summary of the comparison results")


# === Test Cases ===

@pytest.mark.integration
@pytest.mark.skipif(
    not get_openrouter_api_key(), 
    reason="OpenRouter API key not available. Set OPENROUTER_API_KEY environment variable."
)
def test_price_comparison_e2e_with_openrouter():
    """
    End-to-end test for price comparison task using OpenRouter.
    
    Tests that the agent can:
    1. Understand the price comparison goal
    2. Plan to gather prices from both retailers
    3. Execute price fetching and currency conversion
    4. Complete within max iterations
    5. Produce correct final answer
    6. Call each tool the expected number of times
    """
    from test_config import (
        get_openrouter_api_key, 
        OPENROUTER_MODEL, 
        MAX_ITERATIONS, 
        PRODUCT_NAME,
        EXPECTED_CHEAPEST
    )
    
    # Reset all spies
    retail_a_spy.reset()
    retail_b_spy.reset()
    currency_converter_spy.reset()
    
    # Goal definition
    goal = (
        f"Compare the price of '{PRODUCT_NAME}' between Retail A and Retail B. "
        "Get prices from both retailers, convert currencies to USD if needed, "
        "and determine which retailer offers the cheaper price. "
        "Provide a detailed comparison with the price difference."
    )
    
    # Tools available to the agent
    test_tools = {
        "get_retail_a_price": get_retail_a_price_tool,
        "get_retail_b_price": get_retail_b_price_tool,
        "convert_currency": convert_currency_tool,
    }
    
    # Run the agent
    result = run_agent(
        goal=goal,
        model=OPENROUTER_MODEL,
        api_key=get_openrouter_api_key(),
        tools=test_tools,
        output_format=PriceComparisonResult,
        max_iterations=MAX_ITERATIONS,
        verbose=True  # Enable verbose for debugging
    )
    
    # === Assertions ===
    
    # 1. Test completed successfully
    assert result is not None, "Agent should return a result"
    assert result.get("status") in ["completed_with_formatting", "completed_without_formatting"], \
        f"Agent should complete successfully, got status: {result.get('status')}"
    
    # 2. Test completed within max iterations
    iterations_used = result.get("iterations_used", MAX_ITERATIONS + 1)
    assert iterations_used <= MAX_ITERATIONS, \
        f"Agent should complete within {MAX_ITERATIONS} iterations, used {iterations_used}"
    
    # 3. Test that tools were called
    assert retail_a_spy.call_count >= 1, \
        f"Retail A price tool should be called at least once, called {retail_a_spy.call_count} times"
    assert retail_b_spy.call_count >= 1, \
        f"Retail B price tool should be called at least once, called {retail_b_spy.call_count} times"
    assert currency_converter_spy.call_count >= 1, \
        f"Currency converter should be called at least once, called {currency_converter_spy.call_count} times"
    
    # 4. Test tool call efficiency (shouldn't call tools excessively)
    assert retail_a_spy.call_count <= 2, \
        f"Retail A should not be called more than 2 times, called {retail_a_spy.call_count} times"
    assert retail_b_spy.call_count <= 2, \
        f"Retail B should not be called more than 2 times, called {retail_b_spy.call_count} times"
    assert currency_converter_spy.call_count <= 3, \
        f"Currency converter should not be called more than 3 times, called {currency_converter_spy.call_count} times"
    
    # 5. Test structured output if formatting succeeded
    formatted_result = result.get("result")
    if result.get("formatted_output", False) and formatted_result:
        # Validate structured output
        assert hasattr(formatted_result, 'cheapest_retailer'), "Result should have cheapest_retailer field"
        assert hasattr(formatted_result, 'retail_a_price_usd'), "Result should have retail_a_price_usd field"
        assert hasattr(formatted_result, 'retail_b_price_usd'), "Result should have retail_b_price_usd field"
        
        # Validate correctness of comparison
        # Retail A: 10 USD, Retail B: 6 EUR = ~6.60 USD, so Retail B should be cheaper
        assert formatted_result.cheapest_retailer == EXPECTED_CHEAPEST, \
            f"{EXPECTED_CHEAPEST} should be cheaper, but got: {formatted_result.cheapest_retailer}"
        
        # Validate price values are reasonable
        assert 9.5 <= formatted_result.retail_a_price_usd <= 10.5, \
            f"Retail A price should be around 10 USD, got: {formatted_result.retail_a_price_usd}"
        assert 6.0 <= formatted_result.retail_b_price_usd <= 7.0, \
            f"Retail B price should be around 6.60 USD, got: {formatted_result.retail_b_price_usd}"
    
    # 6. Test raw data contains expected information
    raw_data = result.get("raw_data", {})
    assert "retail_a_price" in raw_data, "Raw data should contain Retail A price"
    assert "retail_b_price" in raw_data, "Raw data should contain Retail B price"
    assert "currency_conversion" in raw_data, "Raw data should contain currency conversion"
    
    # 7. Validate specific tool call arguments
    # Check that product name was passed to retail tools
    retail_a_calls = retail_a_spy.calls
    retail_b_calls = retail_b_spy.calls
    
    assert len(retail_a_calls) > 0, "Retail A should have been called"
    assert len(retail_b_calls) > 0, "Retail B should have been called"
    
    # Check conversion was attempted (EUR to USD for Retail B)
    currency_calls = currency_converter_spy.calls
    assert len(currency_calls) > 0, "Currency converter should have been called"
    
    # Find EUR to USD conversion
    eur_to_usd_conversion = None
    for call in currency_calls:
        args = call["args"]
        if args.get("from_currency") == "EUR" and args.get("to_currency") == "USD":
            eur_to_usd_conversion = call
            break
    
    assert eur_to_usd_conversion is not None, \
        "Should have attempted EUR to USD conversion for Retail B price"
    assert eur_to_usd_conversion["args"]["amount"] == 6.0, \
        "Should convert 6.0 EUR from Retail B"
    
    print("\n=== Test Results ===")
    print(f"✓ Completed in {iterations_used}/{MAX_ITERATIONS} iterations")
    print(f"✓ Retail A called {retail_a_spy.call_count} times")
    print(f"✓ Retail B called {retail_b_spy.call_count} times") 
    print(f"✓ Currency converter called {currency_converter_spy.call_count} times")
    print(f"✓ Status: {result.get('status')}")
    print(f"✓ Formatted output: {result.get('formatted_output', False)}")
    
    if formatted_result:
        print(f"✓ Final answer: {formatted_result.cheapest_retailer} is cheaper")
        print(f"✓ Retail A: ${formatted_result.retail_a_price_usd:.2f}")
        print(f"✓ Retail B: ${formatted_result.retail_b_price_usd:.2f}")


@pytest.mark.integration 
def test_price_comparison_e2e_with_mock_llm():
    """
    Alternative test using mock LLM for faster CI/CD testing.
    """
    from src.tagent.llm_adapter import set_llm_adapter, MockLLMAdapter
    
    # Reset spies
    retail_a_spy.reset()
    retail_b_spy.reset() 
    currency_converter_spy.reset()
    
    # Configure mock responses for the agent decision flow
    # Need more responses to handle repeated calls from state machine
    mock_responses = [
        # Responses for decision making (will be repeated as needed)
        '{"action": "execute", "params": {"tool": "get_retail_a_price", "args": {"product": "Wireless Headphones"}}, "reasoning": "Getting price from Retail A"}',
        '{"action": "execute", "params": {"tool": "get_retail_b_price", "args": {"product": "Wireless Headphones"}}, "reasoning": "Getting price from Retail B"}',
        '{"action": "execute", "params": {"tool": "convert_currency", "args": {"amount": 6.0, "from_currency": "EUR", "to_currency": "USD"}}, "reasoning": "Converting EUR to USD"}',
        '{"action": "evaluate", "params": {"achieved": true}, "reasoning": "All prices gathered and converted, comparison complete"}',
        '{"action": "evaluate", "params": {"achieved": true}, "reasoning": "Task completed successfully"}',
        '{"action": "evaluate", "params": {"achieved": true}, "reasoning": "Goal achieved"}',
        '{"action": "evaluate", "params": {"achieved": true}, "reasoning": "Ready to finalize"}',
        '{"action": "evaluate", "params": {"achieved": true}, "reasoning": "Success"}',
    ]
    
    # Set up mock adapter
    mock_adapter = MockLLMAdapter(responses=mock_responses)
    set_llm_adapter(mock_adapter)
    
    try:
        # Test configuration
        goal = "Compare price of 'Wireless Headphones' between Retail A and Retail B with currency conversion"
        test_tools = {
            "get_retail_a_price": get_retail_a_price_tool,
            "get_retail_b_price": get_retail_b_price_tool, 
            "convert_currency": convert_currency_tool,
        }
        
        # Run agent
        result = run_agent(
            goal=goal,
            model="mock-model",
            tools=test_tools,
            output_format=PriceComparisonResult,
            max_iterations=7,
            verbose=False
        )
        
        # Validate results - be more flexible with exact counts
        assert result is not None, "Agent should return a result"
        assert retail_a_spy.call_count >= 1, f"Retail A should be called at least once, got {retail_a_spy.call_count}"
        assert retail_b_spy.call_count >= 1, f"Retail B should be called at least once, got {retail_b_spy.call_count}" 
        assert currency_converter_spy.call_count >= 1, f"Currency converter should be called at least once, got {currency_converter_spy.call_count}"
        
        # Validate that all tools were called
        assert retail_a_spy.call_count > 0
        assert retail_b_spy.call_count > 0
        assert currency_converter_spy.call_count > 0
        
        print("\\n=== Mock Test Results ===") 
        print(f"✓ Tools called correctly")
        print(f"✓ Retail A: {retail_a_spy.call_count} calls")
        print(f"✓ Retail B: {retail_b_spy.call_count} calls")
        print(f"✓ Currency converter: {currency_converter_spy.call_count} calls")
        print(f"✓ Result status: {result.get('status', 'unknown')}")
        
    finally:
        # Reset to default adapter
        from src.tagent.llm_adapter import LiteLLMAdapter
        set_llm_adapter(LiteLLMAdapter())


if __name__ == "__main__":
    # Run the test directly for development
    test_price_comparison_e2e_with_mock_llm()
    print("\n" + "="*50)
    # Uncomment to test with real OpenRouter (requires API key)
    # test_price_comparison_e2e_with_openrouter()
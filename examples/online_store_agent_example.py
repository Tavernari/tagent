import json
import sys
import os
from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel, Field

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tagent.agent import run_agent

# --- 1. Pydantic Model Definition for the Final Output ---
# This model ensures the agent's output is a structured and validated JSON.

class OrderPrediction(BaseModel):
    predicted_cost: float = Field(..., description="The predicted total cost of the order.")
    products: List[Dict[str, Any]] = Field(..., description="A list of products in the order with their details.")
    summary: str = Field(..., description="An AI-generated summary of the order prediction.")

# --- 2. Fake Tool Definitions ---
# Each function is adapted to the TAgent's Store format:
# It receives (state, args) and returns a tuple (key_to_update, value).

def get_available_products_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Retrieves a list of available products in the store.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments (none needed for this tool).
            
    Returns:
        A tuple with ('available_products', products) where products is a list of product names.
    """
    # Fake Data
    products = ["Laptop", "Mouse", "Keyboard"]
    
    # Updates the 'available_products' key in the agent's state
    return ('available_products', products)

def get_stock_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Gets the stock quantity for a specific product.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments.
            - product_name (str): The name of the product to check stock for.
            
    Returns:
        A tuple with ('product_stock', stock_info) where stock_info contains the product name and stock quantity.
        
    Example:
        args = {"product_name": "Laptop"}
    """
    product_name = args.get('product_name')
    if not product_name:
        return None

    # Fake Data
    stock_data = {
        "Laptop": 10,
        "Mouse": 150,
        "Keyboard": 75
    }
    
    stock_info = {
        "product_name": product_name,
        "stock": stock_data.get(product_name, 0)
    }
    
    # Appends to the 'product_stock' list in the state
    current_stock = state.get('product_stock', [])
    current_stock.append(stock_info)
    return ('product_stock', current_stock)

def get_price_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Gets the current purchase price for a specific product.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments.
            - product_name (str): The name of the product to get the price for.
            
    Returns:
        A tuple with ('product_prices', price_info) where price_info contains the product name and price.
        
    Example:
        args = {"product_name": "Laptop"}
    """
    product_name = args.get('product_name')
    if not product_name:
        return None

    # Fake Data
    price_data = {
        "Laptop": 1200.50,
        "Mouse": 25.00,
        "Keyboard": 75.75
    }
    
    price_info = {
        "product_name": product_name,
        "price": price_data.get(product_name, 0.0)
    }
    
    # Appends to the 'product_prices' list in the state
    current_prices = state.get('product_prices', [])
    current_prices.append(price_info)
    return ('product_prices', current_prices)

# --- 3. Agent Configuration and Execution ---

if __name__ == "__main__":
    # The clear goal for the agent
    agent_goal = "First, get all available products in the store. Then, for each product, get its stock and price. Finally, create an order prediction with the total estimated cost and a summary."

    # Dictionary registering the available tools
    agent_tools = {
        "get_available_products": get_available_products_tool,
        "get_stock": get_stock_tool,
        "get_price": get_price_tool,
    }

    print("--- Starting Online Store Agent ---")
    
    # Execute the agent loop
    # The agent will use an LLM (configured in `agent.py`) to decide which tool to call at each step.
    # The `output_format` ensures the final output is an `OrderPrediction` object.
    final_output = run_agent(
        goal=agent_goal,
        model="openrouter/google/gemma-3-27b-it", # Model the agent will use for decisions
        tools=agent_tools,
        output_format=OrderPrediction,
        verbose=True,
    )

    print("--- Final Agent Result ---")
    if final_output:
        # Check for chat history in the result
        if isinstance(final_output, dict) and 'chat_summary' in final_output:
            print("\n" + final_output['chat_summary'])
            
            # Check the execution status
            status = final_output.get('status', 'unknown')
            print(f"\n--- STATUS: {status.upper().replace('_', ' ')} ---")
            
            if final_output['result']:
                print("\n--- STRUCTURED RESULT ---")
                # The Pydantic model can be easily converted to a JSON
                json_output = final_output['result'].model_dump_json(indent=4)
                print(json_output)
            elif final_output.get('raw_data'):
                print("\n--- COLLECTED DATA (UNFORMATTED) ---")
                # Show raw collected data
                raw_data = final_output['raw_data']
                collected_data = {k: v for k, v in raw_data.items() 
                                if k not in ['goal', 'achieved', 'used_tools'] and v}
                json_output = json.dumps(collected_data, indent=4, ensure_ascii=False)
                print(json_output)
                
                if final_output.get('error'):
                    print(f"\n⚠️  Warning: {final_output['error']}")
            else:
                print(f"\nError: {final_output.get('error', 'Result not available')}")
        else:
            # Old result without chat
            json_output = final_output.model_dump_json(indent=4)
            print(json_output)
    else:
        print("The agent could not generate a final output.")

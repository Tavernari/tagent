

"""
Online Store Tools for TAgent CLI Example
"""

from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel, Field

# --- Data Models ---
class Product(BaseModel):
    name: str
    price: float
    stock: int

# --- Tool Functions ---

def get_available_products_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Retrieves a list of available products in the store.
    """
    products = [
        Product(name="Laptop", price=1200.50, stock=10),
        Product(name="Mouse", price=25.00, stock=150),
        Product(name="Keyboard", price=75.75, stock=75)
    ]
    return ('available_products', [p.model_dump() for p in products])

def get_product_details_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Gets the details (stock and price) for a specific product.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments.
            - product_name (str): The name of the product to get details for.
            
    Returns:
        A tuple with ('product_details', details_info) where details_info contains the product name, price, and stock.
    """
    product_name = args.get('product_name')
    if not product_name:
        return None

    all_products = state.get('available_products', [])
    
    product_info = next((p for p in all_products if p['name'].lower() == product_name.lower()), None)
    
    if product_info:
        current_details = state.get('product_details', [])
        current_details.append(product_info)
        return ('product_details', current_details)
    else:
        return None

def calculate_order_cost_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Calculates the total estimated cost of an order based on selected products.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments.
            - products (List[Dict[str, Any]]): A list of products with 'name' and 'quantity'.
            
    Returns:
        A tuple with ('order_cost', cost_info) where cost_info contains the total cost and a list of products with their calculated prices.
    """
    products_to_order = args.get('products', [])
    if not products_to_order:
        return None

    all_available_products = state.get('available_products', [])
    total_cost = 0.0
    ordered_products_details = []

    for item in products_to_order:
        product_name = item.get('name')
        quantity = item.get('quantity', 1)
        
        product_data = next((p for p in all_available_products if p['name'].lower() == product_name.lower()), None)
        
        if product_data:
            price = product_data.get('price', 0.0)
            cost = price * quantity
            total_cost += cost
            ordered_products_details.append({
                "name": product_name,
                "quantity": quantity,
                "unit_price": price,
                "total_price": cost
            })
    
    return ('order_cost', {
        "total_estimated_cost": total_cost,
        "products": ordered_products_details
    })

def generate_order_summary_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Generates a summary of the predicted order.
    """
    order_cost_info = state.get('order_cost', {})
    total_cost = order_cost_info.get('total_estimated_cost', 0.0)
    products_in_order = order_cost_info.get('products', [])

    summary = f"Predicted Order Summary:\n"
    if products_in_order:
        summary += "Products:\n"
        for p in products_in_order:
            summary += f"  - {p.get('name')} (Quantity: {p.get('quantity')}, Unit Price: ${p.get('unit_price'):.2f}, Total: ${p.get('total_price'):.2f})\n"
        summary += f"\nTotal Estimated Cost: ${total_cost:.2f}\n"
    else:
        summary += "No products in the predicted order.\n"
    
    return ('order_summary', summary)

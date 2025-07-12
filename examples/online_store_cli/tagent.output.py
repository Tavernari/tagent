
"""
Online Store Output Schema for TAgent CLI Example
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any

class OrderPrediction(BaseModel):
    predicted_cost: float = Field(..., description="The predicted total cost of the order.")
    products: List[Dict[str, Any]] = Field(..., description="A list of products in the order with their details.")
    summary: str = Field(..., description="An AI-generated summary of the order prediction.")

# This is the variable that main.py will look for
output_schema = OrderPrediction

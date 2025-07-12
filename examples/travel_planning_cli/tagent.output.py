"""
Travel Planning Output Schema for TAgent CLI Example
Based on the original travel_planning_agent_example.py output format
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any

class FlightOption(BaseModel):
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    price: float
    layovers: int = Field(0, description="Number of layovers.")
    duration: str = Field("", description="Flight duration, e.g., '4h'.")

class HotelOption(BaseModel):
    name: str
    location: str
    price_per_night: float
    rating: float
    amenities: List[str] = Field([], description="List of amenities like 'free wifi', 'pool'.")

class ActivityOption(BaseModel):
    name: str
    description: str
    estimated_cost: float
    duration: str = Field("", description="Estimated duration, e.g., '2 hours'.")
    best_time: str = Field("", description="Best time to do the activity, e.g., 'morning'.")

class TravelPlan(BaseModel):
    """Complete travel plan with flights, hotels, activities and budget breakdown."""
    
    destination: str = Field(..., description="The planned travel destination.")
    travel_dates: str = Field(..., description="The dates for the trip (e.g., '2025-08-01 to 2025-08-07').")
    budget: float = Field(..., description="The specified budget for the trip.")
    
    # Flight information
    flight_data: Dict[str, Any] = Field(
        {'options': [], 'status': 'Not searched.'}, 
        description="Dictionary containing flight options and search status."
    )
    
    # Hotel information
    hotels: List[HotelOption] = Field([], description="List of selected hotel options.")
    
    # Activities
    activities: List[ActivityOption] = Field([], description="List of selected activity options.")
    
    # Cost breakdown
    total_estimated_cost: float = Field(0.0, description="The total estimated cost of the entire trip.")
    
    # Summary
    itinerary_summary: str = Field(..., description="A detailed summary of the travel itinerary.")

# This is the variable that main.py will look for
output_schema = TravelPlan
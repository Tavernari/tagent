"""
Travel Output Schema for TAgent
Defines the expected output format for travel planning agents.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FlightInfo(BaseModel):
    airline: str = Field(..., description="Airline name")
    flight_number: str = Field(..., description="Flight number")
    origin: str = Field(..., description="Departure city")
    destination: str = Field(..., description="Arrival city")
    departure: str = Field(..., description="Departure date and time")
    arrival: str = Field(..., description="Arrival date and time")
    price: float = Field(..., description="Flight price in USD")
    duration: str = Field(..., description="Flight duration")

class HotelInfo(BaseModel):
    name: str = Field(..., description="Hotel name")
    location: str = Field(..., description="Hotel location")
    price_per_night: float = Field(..., description="Price per night in USD")
    rating: float = Field(..., description="Hotel rating (0-5)")
    amenities: List[str] = Field(default=[], description="Hotel amenities")

class ActivityInfo(BaseModel):
    name: str = Field(..., description="Activity name")
    category: str = Field(..., description="Activity category")
    duration: str = Field(..., description="Activity duration")
    price: float = Field(..., description="Activity price in USD")
    rating: float = Field(..., description="Activity rating (0-5)")
    description: str = Field(..., description="Activity description")

class BudgetBreakdown(BaseModel):
    flights: Optional[float] = Field(None, description="Total flight costs")
    hotels: Optional[float] = Field(None, description="Total hotel costs")
    activities: Optional[float] = Field(None, description="Total activity costs")
    hotel_per_night: Optional[float] = Field(None, description="Hotel cost per night")

class TravelPlan(BaseModel):
    """Complete travel plan with flights, hotels, activities and budget."""
    
    destination: str = Field(..., description="Travel destination")
    travel_dates: str = Field(..., description="Travel dates (YYYY-MM-DD to YYYY-MM-DD)")
    
    # Flight information
    recommended_flight: Optional[FlightInfo] = Field(None, description="Best recommended flight option")
    alternative_flights: List[FlightInfo] = Field(default=[], description="Alternative flight options")
    
    # Hotel information
    recommended_hotel: Optional[HotelInfo] = Field(None, description="Best recommended hotel")
    alternative_hotels: List[HotelInfo] = Field(default=[], description="Alternative hotel options")
    
    # Activities
    recommended_activities: List[ActivityInfo] = Field(default=[], description="Recommended activities")
    
    # Budget calculation
    total_budget: float = Field(..., description="Total estimated trip cost in USD")
    budget_breakdown: BudgetBreakdown = Field(..., description="Detailed budget breakdown")
    
    # Additional information
    trip_summary: str = Field(..., description="Summary of the complete travel plan")
    recommendations: List[str] = Field(default=[], description="Additional travel recommendations")
    
    # Search metadata
    search_success: bool = Field(True, description="Whether all searches were successful")
    search_errors: List[str] = Field(default=[], description="Any errors encountered during search")

# This is the variable that main.py will look for
output_schema = TravelPlan
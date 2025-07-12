import json
import sys
import os
from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel, Field

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tagent.agent import run_agent

# --- 1. Pydantic Models for Agent Output and Data Structures ---

class FlightOption(BaseModel):
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    price: float

class HotelOption(BaseModel):
    name: str
    location: str
    price_per_night: float
    rating: float

class ActivityOption(BaseModel):
    name: str
    description: str
    estimated_cost: float

class TravelPlan(BaseModel):
    destination: str = Field(..., description="The planned travel destination.")
    travel_dates: str = Field(..., description="The dates for the trip (e.g., '2025-08-01 to 2025-08-07').")
    budget: float = Field(..., description="The specified budget for the trip.")
    flights: List[FlightOption] = Field([], description="List of selected flight options.")
    flight_search_status: str = Field("", description="Status of the flight search (e.g., 'Flights found successfully.', 'No flights found for the route.', 'No flights found within the budget.').")
    hotels: List[HotelOption] = Field([], description="List of selected hotel options.")
    activities: List[ActivityOption] = Field([], description="List of selected activity options.")
    total_estimated_cost: float = Field(0.0, description="The total estimated cost of the entire trip.")
    itinerary_summary: str = Field(..., description="A detailed summary of the travel itinerary.")

# --- 2. Fake Tool Definitions ---
# Each function is adapted to the TAgent's Store format:
# It receives (state, args) and returns a tuple (key_to_update, value).

def search_flights_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for flight options based on origin, destination, dates, and budget.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments:
            - origin (str): Departure city.
            - destination (str): Arrival city.
            - dates (str): Travel dates (e.g., '2025-08-01 to 2025-08-07').
            - budget (float): Maximum budget for flights.
            
    Returns:
        A tuple with ('flight_options', List[FlightOption]) where flight_options are available flights.
    """
    origin = state.get('origin')
    destination = state.get('destination')
    dates = state.get('travel_dates')
    budget = state.get('budget')

    if not all([origin, destination, dates, budget]):
        return None

    # Fake Data - Expanded and filtered by origin, destination, and budget
    all_flights = [
        FlightOption(airline="AirLines", flight_number="AL101", origin="New York", destination="Paris", departure_time="08:00", arrival_time="12:00", price=350.00),
        FlightOption(airline="GlobalAir", flight_number="GA205", origin="London", destination="Rome", departure_time="10:30", arrival_time="14:30", price=420.00),
        FlightOption(airline="FlyFast", flight_number="FF300", origin="Berlin", destination="Madrid", departure_time="14:00", arrival_time="18:00", price=280.00),
        FlightOption(airline="Oceanic", flight_number="OC400", origin="Tokyo", destination="Sydney", departure_time="09:00", arrival_time="13:00", price=900.00),
        FlightOption(airline="StarLink", flight_number="SL501", origin="New York", destination="London", departure_time="11:00", arrival_time="15:00", price=380.00),
        FlightOption(airline="AirLines", flight_number="AL102", origin="Paris", destination="New York", departure_time="07:00", arrival_time="11:00", price=320.00),
        FlightOption(airline="GlobalAir", flight_number="GA206", origin="Rome", destination="London", departure_time="13:00", arrival_time="17:00", price=450.00),
        FlightOption(airline="FlyFast", flight_number="FF301", origin="Madrid", destination="Berlin", departure_time="16:00", arrival_time="20:00", price=290.00),
        FlightOption(airline="Oceanic", flight_number="OC401", origin="Sydney", destination="Tokyo", departure_time="10:00", arrival_time="14:00", price=950.00),
        FlightOption(airline="StarLink", flight_number="SL502", origin="London", destination="New York", departure_time="12:00", arrival_time="16:00", price=390.00),
        FlightOption(airline="AirLines", flight_number="AL103", origin="New York", destination="Paris", departure_time="09:30", arrival_time="13:30", price=360.00),
        FlightOption(airline="GlobalAir", flight_number="GA207", origin="London", destination="Rome", departure_time="15:00", arrival_time="19:00", price=430.00),
        FlightOption(airline="FlyFast", flight_number="FF302", origin="Berlin", destination="Madrid", departure_time="18:00", arrival_time="22:00", price=300.00),
        FlightOption(airline="Oceanic", flight_number="OC402", origin="Tokyo", destination="Sydney", departure_time="11:30", arrival_time="15:30", price=910.00),
        FlightOption(airline="StarLink", flight_number="SL503", origin="New York", destination="London", departure_time="13:30", arrival_time="17:30", price=400.00),
        FlightOption(airline="AirLines", flight_number="AL104", origin="Paris", destination="New York", departure_time="06:00", arrival_time="10:00", price=310.00),
        FlightOption(airline="GlobalAir", flight_number="GA208", origin="Rome", destination="London", departure_time="14:00", arrival_time="18:00", price=460.00),
        FlightOption(airline="FlyFast", flight_number="FF303", origin="Madrid", destination="Berlin", departure_time="17:00", arrival_time="21:00", price=270.00),
        FlightOption(airline="Oceanic", flight_number="OC403", origin="Sydney", destination="Tokyo", departure_time="12:30", arrival_time="16:30", price=930.00),
        FlightOption(airline="StarLink", flight_number="SL504", origin="London", destination="New York", departure_time="14:30", arrival_time="18:30", price=410.00),
        FlightOption(airline="AirLines", flight_number="AL105", origin="New York", destination="Paris", departure_time="08:30", arrival_time="12:30", price=340.00),
        FlightOption(airline="GlobalAir", flight_number="GA209", origin="London", destination="Rome", departure_time="10:00", arrival_time="14:00", price=410.00),
        FlightOption(airline="FlyFast", flight_number="FF304", origin="Berlin", destination="Madrid", departure_time="15:00", arrival_time="19:00", price=260.00),
        FlightOption(airline="Oceanic", flight_number="OC404", origin="Tokyo", destination="Sydney", departure_time="09:30", arrival_time="13:30", price=890.00),
        FlightOption(airline="StarLink", flight_number="SL505", origin="New York", destination="London", departure_time="11:30", arrival_time="15:30", price=370.00),
        FlightOption(airline="AirLines", flight_number="AL106", origin="Paris", destination="New York", departure_time="07:30", arrival_time="11:30", price=330.00),
        FlightOption(airline="GlobalAir", flight_number="GA210", origin="Rome", destination="London", departure_time="11:00", arrival_time="15:00", price=400.00),
        FlightOption(airline="FlyFast", flight_number="FF305", origin="Madrid", destination="Berlin", departure_time="13:00", arrival_time="17:00", price=250.00),
        FlightOption(airline="Oceanic", flight_number="OC405", origin="Sydney", destination="Tokyo", departure_time="08:30", arrival_time="12:30", price=880.00),
        FlightOption(airline="StarLink", flight_number="SL506", origin="London", destination="New York", departure_time="10:30", arrival_time="14:30", price=360.00)
    ]
    
    # Filter by origin and destination first
    route_flights = [
        f for f in all_flights 
        if f.origin.lower() == origin.lower() and f.destination.lower() == destination.lower()
    ]
    
    if not route_flights:
        return ('flight_options', []), ('flight_search_status', f"No flights found for the route {origin} to {destination}.")
    
    # Then filter by budget
    affordable_flights = [f for f in route_flights if f.price <= budget]
    
    if not affordable_flights:
        return ('flight_options', []), ('flight_search_status', f"No flights found within the budget of ${budget:.2f} for the route {origin} to {destination}.")
    
    return ('flight_options', [f.model_dump() for f in affordable_flights]), ('flight_search_status', "Flights found successfully.")

def search_hotels_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for hotel options based on destination, dates, budget, and preferences.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments:
            - destination (str): City for hotels.
            - dates (str): Check-in/check-out dates.
            - budget (float): Maximum budget per night.
            - preferences (str): e.g., 'luxury', 'budget', 'near beach'.
            
    Returns:
        A tuple with ('hotel_options', List[HotelOption]) where hotel_options are available hotels.
    """
    destination = state.get('destination')
    dates = state.get('travel_dates')
    budget = state.get('budget')
    preferences = state.get('hotel_preferences', '').lower()

    if not all([destination, dates, budget]):
        return None

    # Fake Data - Filtered by budget, preferences, and destination
    all_hotels = [
        HotelOption(name="Grand Plaza Paris", location="Paris Downtown", price_per_night=250.00, rating=4.5),
        HotelOption(name="Budget Inn Paris", location="Paris Suburb", price_per_night=80.00, rating=3.0),
        HotelOption(name="Beach Resort Paris", location="Paris Coastal Area", price_per_night=350.00, rating=4.8),
        HotelOption(name="Rome Luxury Suites", location="Rome City Center", price_per_night=300.00, rating=4.7),
        HotelOption(name="Rome Budget Stay", location="Rome Termini", price_per_night=70.00, rating=2.9),
        HotelOption(name="London Central Hotel", location="London Westminster", price_per_night=280.00, rating=4.6),
        HotelOption(name="London Budget Hostel", location="London East End", price_per_night=60.00, rating=2.5),
        HotelOption(name="Berlin Grand Hotel", location="Berlin Mitte", price_per_night=220.00, rating=4.3),
        HotelOption(name="Berlin Hostel", location="Berlin Kreuzberg", price_per_night=50.00, rating=2.8),
        HotelOption(name="Madrid City Center", location="Madrid Sol", price_per_night=200.00, rating=4.2),
        HotelOption(name="Madrid Budget Inn", location="Madrid Latina", price_per_night=45.00, rating=2.7),
        HotelOption(name="Tokyo Imperial Hotel", location="Tokyo Chiyoda", price_per_night=400.00, rating=4.9),
        HotelOption(name="Tokyo Capsule Hotel", location="Tokyo Shinjuku", price_per_night=30.00, rating=3.5),
        HotelOption(name="Sydney Harbour View", location="Sydney Circular Quay", price_per_night=380.00, rating=4.8),
        HotelOption(name="Sydney Backpackers", location="Sydney Kings Cross", price_per_night=40.00, rating=3.2)
    ]
    
    filtered_hotels = []
    for h in all_hotels:
        if h.location.lower().startswith(destination.lower()) and h.price_per_night <= budget:
            if "luxury" in preferences and h.rating >= 4.0:
                filtered_hotels.append(h)
            elif "budget" in preferences and h.price_per_night <= 100.00:
                filtered_hotels.append(h)
            elif not preferences: # No specific preference
                filtered_hotels.append(h)

    return ('hotel_options', [h.model_dump() for h in filtered_hotels])

def search_activities_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for activities based on destination, dates, and interests.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments:
            - destination (str): City for activities.
            - dates (str): Dates for activities.
            - interests (str): e.g., 'museums', 'adventure', 'food'.
            
    Returns:
        A tuple with ('activity_options', List[ActivityOption]) where activity_options are available activities.
    """
    destination = state.get('destination')
    dates = state.get('travel_dates')
    interests = state.get('activity_interests', '').lower()

    if not all([destination, dates]):
        return None

    # Fake Data - Filtered by interests and destination
    all_activities = [
        ActivityOption(name="Paris City Museum Tour", description="Explore Parisian history and art.", estimated_cost=30.00),
        ActivityOption(name="Paris Seine River Cruise", description="Enjoy a scenic cruise on the Seine.", estimated_cost=45.00),
        ActivityOption(name="Paris Food Tasting Tour", description="Sample traditional French cuisine.", estimated_cost=70.00),
        ActivityOption(name="Rome Colosseum Tour", description="Discover ancient Roman history.", estimated_cost=40.00),
        ActivityOption(name="Rome Vatican City Tour", description="Visit the Vatican Museums and St. Peter's Basilica.", estimated_cost=60.00),
        ActivityOption(name="Rome Pasta Making Class", description="Learn to make authentic Italian pasta.", estimated_cost=80.00),
        ActivityOption(name="London Tower Bridge Experience", description="Explore the iconic Tower Bridge.", estimated_cost=35.00),
        ActivityOption(name="London British Museum Visit", description="Discover world history and culture.", estimated_cost=0.00),
        ActivityOption(name="London West End Show", description="Enjoy a world-class theater performance.", estimated_cost=100.00),
        ActivityOption(name="Berlin Wall Memorial", description="Learn about the history of the Berlin Wall.", estimated_cost=0.00),
        ActivityOption(name="Berlin Museum Island Tour", description="Visit a complex of five world-renowned museums.", estimated_cost=25.00),
        ActivityOption(name="Berlin Nightlife Tour", description="Experience Berlin's vibrant nightlife.", estimated_cost=50.00),
        ActivityOption(name="Madrid Prado Museum Tour", description="Admire masterpieces of European art.", estimated_cost=20.00),
        ActivityOption(name="Madrid Royal Palace Visit", description="Explore the official residence of the Spanish Royal Family.", estimated_cost=30.00),
        ActivityOption(name="Madrid Tapas Tour", description="Savor traditional Spanish tapas.", estimated_cost=60.00),
        ActivityOption(name="Tokyo Imperial Palace Gardens", description="Stroll through beautiful gardens.", estimated_cost=0.00),
        ActivityOption(name="Tokyo Shibuya Crossing Experience", description="Witness the famous Shibuya scramble.", estimated_cost=0.00),
        ActivityOption(name="Tokyo Robot Restaurant Show", description="Enjoy a unique and eccentric show.", estimated_cost=150.00),
        ActivityOption(name="Sydney Opera House Tour", description="Discover the iconic Sydney Opera House.", estimated_cost=40.00),
        ActivityOption(name="Sydney Harbour Bridge Climb", description="Climb the famous Harbour Bridge for panoramic views.", estimated_cost=200.00),
        ActivityOption(name="Sydney Bondi Beach Surfing Lesson", description="Learn to surf at Bondi Beach.", estimated_cost=90.00)
    ]
    
    filtered_activities = []
    for a in all_activities:
        if destination.lower() in a.name.lower() or destination.lower() in a.description.lower():
            if "museums" in interests and ("museum" in a.name.lower() or "museum" in a.description.lower()):
                filtered_activities.append(a)
            elif "adventure" in interests and ("adventure" in a.name.lower() or "adventure" in a.description.lower() or "climb" in a.name.lower() or "hike" in a.name.lower()):
                filtered_activities.append(a)
            elif "food" in interests and ("food" in a.name.lower() or "food" in a.description.lower() or "tasting" in a.name.lower() or "cooking" in a.name.lower()):
                filtered_activities.append(a)
            elif "history" in interests and ("history" in a.name.lower() or "history" in a.description.lower() or "palace" in a.name.lower() or "colosseum" in a.name.lower() or "wall" in a.name.lower()):
                filtered_activities.append(a)
            elif not interests: # No specific interest
                filtered_activities.append(a)

    return ('activity_options', [a.model_dump() for a in filtered_activities])

def calculate_total_cost_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Calculates the total estimated cost based on selected flights, hotels, and activities.
    
    Args:
        state: Current agent state (expects 'flight_options', 'hotel_options', 'activity_options' to be present).
        args: Dictionary with tool arguments (none needed, uses state).
            
    Returns:
        A tuple with ('total_estimated_cost', float).
    """
    total_cost = 0.0
    
    flights = state.get('flight_options', [])
    for f in flights:
        total_cost += f['price']
        
    hotels = state.get('hotel_options', [])
    # Assuming a fixed number of nights for hotel cost calculation for simplicity
    num_nights = 6 # Example: 6 nights for a 7-day trip
    for h in hotels:
        total_cost += h['price_per_night'] * num_nights
        
    activities = state.get('activity_options', [])
    for a in activities:
        total_cost += a['estimated_cost']
        
    return ('total_estimated_cost', total_cost)

def generate_itinerary_summary_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Generates a summary of the travel itinerary based on selected flights, hotels, and activities.
    
    Args:
        state: Current agent state (expects 'flight_options', 'hotel_options', 'activity_options' to be present).
        args: Dictionary with tool arguments (none needed, uses state).
            
    Returns:
        A tuple with ('itinerary_summary', str).
    """
    destination = state.get('destination', 'Unknown Destination')
    travel_dates = state.get('travel_dates', 'Unknown Dates')
    flights = state.get('flight_options', [])
    hotels = state.get('hotel_options', [])
    activities = state.get('activity_options', [])
    total_cost = state.get('total_estimated_cost', 0.0)

    flight_search_status = state.get('flight_search_status', 'Not searched.')

    summary = f"Travel Plan for {destination} from {travel_dates}:\n\n"
    
    summary += f"Flight Search Status: {flight_search_status}\n"

    if flights:
        summary += "Flights:\n"
        for f in flights:
            summary += f"  - {f['airline']} {f['flight_number']} ({f['origin']} to {f['destination']}): {f['departure_time']} to {f['arrival_time']} (${f['price']:.2f})\n"
    
    if hotels:
        summary += "\nHotels:\n"
        for h in hotels:
            summary += f"  - {h['name']} ({h['location']}): ${h['price_per_night']:.2f}/night (Rating: {h['rating']})\n"
            
    if activities:
        summary += "\nActivities:\n"
        for a in activities:
            summary += f"  - {a['name']}: {a['description']} (Est. Cost: ${a['estimated_cost']:.2f})\n"
            
    summary += f"\nTotal Estimated Cost: ${total_cost:.2f}\n"
    summary += "\nThis plan provides a comprehensive overview of your trip, including transportation, accommodation, and leisure activities."
    
    return ('itinerary_summary', summary)

# --- 3. Agent Configuration and Execution ---

if __name__ == "__main__":
    # Define the travel parameters
    travel_destination = "Rome"
    travel_origin = "London"
    travel_dates = "2025-09-10 to 2025-09-17"
    travel_budget = 1500.00 # Total budget for flights, hotels, and activities
    hotel_preferences = "luxury"
    activity_interests = "history, food"

    # The clear and complex goal for the agent
    agent_goal = (
        f"Plan a trip to {travel_destination} from {travel_origin} between {travel_dates}, "
        f"with a total budget of ${travel_budget:.2f}. "
        f"First, find suitable flight options. Then, search for hotel options in {travel_destination} "
        f"with a preference for '{hotel_preferences}' and within the budget. "
        f"Next, find activities in {travel_destination} related to '{activity_interests}'. "
        f"Finally, calculate the total estimated cost of the entire trip and generate a detailed itinerary summary."
    )

    # Dictionary registering the available tools
    agent_tools = {
        "search_flights": search_flights_tool,
        "search_hotels": search_hotels_tool,
        "search_activities": search_activities_tool,
        "calculate_total_cost": calculate_total_cost_tool,
        "generate_itinerary_summary": generate_itinerary_summary_tool,
    }

    print("--- Starting Travel Planning Agent ---")
    
    # Execute the agent loop
    # The agent will use an LLM (configured in `agent.py`) to decide which tool to call at each step.
    # The `output_format` ensures the final output is a `TravelPlan` object.
    final_output = run_agent(
        goal=agent_goal,
        model="openrouter/google/gemma-3-27b-it", # Model the agent will use for decisions
        tools=agent_tools,
        output_format=TravelPlan
    )

    print("\n--- Final Agent Result ---")
    if final_output:
        if isinstance(final_output, dict) and 'chat_summary' in final_output:
            print("\n" + final_output['chat_summary'])
            
            status = final_output.get('status', 'unknown')
            print(f"\n--- STATUS: {status.upper().replace('_', ' ')} ---")
            
            if final_output['result']:
                print("\n--- STRUCTURED RESULT ---")
                json_output = final_output['result'].model_dump_json(indent=4)
                print(json_output)
            elif final_output.get('raw_data'):
                print("\n--- COLLECTED DATA (UNFORMATTED) ---")
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
            json_output = final_output.model_dump_json(indent=4)
            print(json_output)
    else:
        print("The agent could not generate a final output.")
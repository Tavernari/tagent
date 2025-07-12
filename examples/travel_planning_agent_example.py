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
    departure_time: str  # Now includes date, e.g., '2025-08-01 08:00'
    arrival_time: str    # Now includes date, e.g., '2025-08-01 12:00'
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
    destination: str = Field(..., description="The planned travel destination.")
    travel_dates: str = Field(..., description="The dates for the trip (e.g., '2025-08-01 to 2025-08-07').")
    budget: float = Field(..., description="The specified budget for the trip.")
    flight_data: Dict[str, Any] = Field({'options': [], 'status': 'Not searched.'}, description="Dictionary containing flight options and search status.")
    hotels: List[HotelOption] = Field([], description="List of selected hotel options.")
    activities: List[ActivityOption] = Field([], description="List of selected activity options.")
    total_estimated_cost: float = Field(0.0, description="The total estimated cost of the entire trip.")
    itinerary_summary: str = Field(..., description="A detailed summary of the travel itinerary.")

# --- 2. Fake Tool Definitions ---
# Each function is adapted to the TAgent's Store format:
# It receives (state, args) and returns a tuple (key_to_update, value) or a list of such tuples for multiple updates.

def search_flights_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[List[Tuple[str, Any]]]:
    """
    Searches for flight options based on origin, destination, dates, and budget. Simulates real API by filtering on exact dates and adding realistic variations like layovers.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments:
            - origin (str): Departure city.
            - destination (str): Arrival city.
            - dates (str): Travel dates (e.g., '2025-08-01 to 2025-08-07').
            - budget (float): Maximum budget for flights.
            
    Returns:
        A list of tuples like [('flight_options', List[FlightOption]), ('flight_search_status', str)] for multiple updates.
    """
    origin = state.get('origin')
    destination = state.get('destination')
    dates = state.get('travel_dates')
    budget = state.get('budget')

    if not all([origin, destination, dates, budget]):
        return None

    # Parse dates for realistic filtering (assume format 'YYYY-MM-DD to YYYY-MM-DD')
    try:
        dep_date, ret_date = [d.strip() for d in dates.split(' to ')]
    except ValueError:
        return [('flight_search_status', "Invalid date format. Unable to search flights.")]

    # Fake Data - More realistic with dates, layovers, durations, and varied options
    all_flights = [
        FlightOption(airline="Air France", flight_number="AF101", origin="New York", destination="Paris", departure_time=f"{dep_date} 08:00", arrival_time=f"{dep_date} 20:00", price=550.00, layovers=0, duration="12h"),
        FlightOption(airline="Delta", flight_number="DL205", origin="London", destination="Rome", departure_time=f"{dep_date} 10:30", arrival_time=f"{dep_date} 13:00", price=220.00, layovers=0, duration="2h 30m"),
        FlightOption(airline="Lufthansa", flight_number="LH300", origin="Berlin", destination="Madrid", departure_time=f"{dep_date} 14:00", arrival_time=f"{dep_date} 17:00", price=180.00, layovers=0, duration="3h"),
        FlightOption(airline="Qantas", flight_number="QF400", origin="Tokyo", destination="Sydney", departure_time=f"{dep_date} 09:00", arrival_time=f"{dep_date} 19:00", price=700.00, layovers=0, duration="10h"),
        FlightOption(airline="British Airways", flight_number="BA501", origin="New York", destination="London", departure_time=f"{dep_date} 11:00", arrival_time=f"{dep_date} 22:00", price=450.00, layovers=0, duration="11h"),
        FlightOption(airline="Air France", flight_number="AF102", origin="Paris", destination="New York", departure_time=f"{ret_date} 07:00", arrival_time=f"{ret_date} 10:00", price=520.00, layovers=1, duration="9h (1 layover)"),
        FlightOption(airline="Alitalia", flight_number="AZ206", origin="Rome", destination="London", departure_time=f"{ret_date} 13:00", arrival_time=f"{ret_date} 15:30", price=250.00, layovers=0, duration="2h 30m"),
        FlightOption(airline="Iberia", flight_number="IB301", origin="Madrid", destination="Berlin", departure_time=f"{ret_date} 16:00", arrival_time=f"{ret_date} 19:00", price=190.00, layovers=0, duration="3h"),
        FlightOption(airline="JAL", flight_number="JL401", origin="Sydney", destination="Tokyo", departure_time=f"{ret_date} 10:00", arrival_time=f"{ret_date} 18:00", price=750.00, layovers=0, duration="8h"),
        FlightOption(airline="American Airlines", flight_number="AA502", origin="London", destination="New York", departure_time=f"{ret_date} 12:00", arrival_time=f"{ret_date} 15:00", price=460.00, layovers=1, duration="9h (1 layover)"),
        # Add more for variety and realism, including some that might not match dates
        FlightOption(airline="EasyJet", flight_number="EJ103", origin="London", destination="Rome", departure_time=f"2025-09-11 09:00", arrival_time=f"2025-09-11 11:30", price=150.00, layovers=0, duration="2h 30m"),
        FlightOption(airline="Ryanair", flight_number="RY104", origin="London", destination="Rome", departure_time=f"{dep_date} 06:00", arrival_time=f"{dep_date} 08:30", price=100.00, layovers=0, duration="2h 30m"),
        FlightOption(airline="United", flight_number="UA105", origin="New York", destination="Paris", departure_time=f"{dep_date} 18:00", arrival_time=f"{dep_date+1} 08:00", price=600.00, layovers=0, duration="14h"),  # Overnight
    ]
    
    # Realistic filter: by origin, destination, and exact departure date (for outbound; simplify for return)
    route_flights = [
        f for f in all_flights 
        if f.origin.lower() == origin.lower() and f.destination.lower() == destination.lower() and dep_date in f.departure_time
    ]
    
    if not route_flights:
        return ('flight_data', {'options': [], 'status': f"No flights found for the route {origin} to {destination} on {dep_date}."})
    
    # Filter by budget, simulating real affordability check
    affordable_flights = [f for f in route_flights if f.price <= budget]
    
    if not affordable_flights:
        return ('flight_data', {'options': [], 'status': f"No flights found within the budget of ${budget:.2f} for the route {origin} to {destination} on {dep_date}."})
    
    # Return top 3 cheapest for realism
    affordable_flights.sort(key=lambda x: x.price)
    return ('flight_data', {'options': [f.model_dump() for f in affordable_flights[:3]], 'status': "Flights found successfully. Showing top 3 options."})

def search_hotels_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for hotel options based on destination, dates, budget, and preferences. Calculates nights realistically from dates.
    
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
    budget = state.get('budget', 0) / 7  # Approximate per night budget, assuming week-long trip for realism
    preferences = state.get('hotel_preferences', '').lower()

    if not all([destination, dates]):
        return None

    # Parse dates to calculate number of nights (for future cost calc, but here just validate)
    try:
        from datetime import datetime
        dep_date, ret_date = [datetime.strptime(d.strip(), '%Y-%m-%d') for d in dates.split(' to ')]
        num_nights = (ret_date - dep_date).days
        if num_nights <= 0:
            return ('hotel_options', [])
    except ValueError:
        num_nights = 6  # Fallback

    # Fake Data - More realistic with amenities and filtered strictly
    all_hotels = [
        HotelOption(name="Four Seasons Paris", location="Paris Champs-Élysées", price_per_night=450.00, rating=4.8, amenities=["free wifi", "pool", "spa"]),
        HotelOption(name="Ibis Paris", location="Paris Suburb", price_per_night=90.00, rating=3.5, amenities=["free wifi", "breakfast"]),
        HotelOption(name="Ritz Carlton Rome", location="Rome City Center", price_per_night=350.00, rating=4.7, amenities=["free wifi", "pool", "gym"]),
        HotelOption(name="Hostel Roma", location="Rome Termini", price_per_night=50.00, rating=3.0, amenities=["free wifi"]),
        HotelOption(name="The Savoy London", location="London Westminster", price_per_night=400.00, rating=4.9, amenities=["free wifi", "spa", "restaurant"]),
        HotelOption(name="Premier Inn London", location="London East End", price_per_night=80.00, rating=3.8, amenities=["free wifi", "breakfast"]),
        HotelOption(name="Hotel Adlon Berlin", location="Berlin Mitte", price_per_night=300.00, rating=4.6, amenities=["free wifi", "pool", "bar"]),
        HotelOption(name="A&O Hostel Berlin", location="Berlin Kreuzberg", price_per_night=40.00, rating=3.2, amenities=["free wifi"]),
        HotelOption(name="Palace Hotel Madrid", location="Madrid Sol", price_per_night=250.00, rating=4.4, amenities=["free wifi", "gym"]),
        HotelOption(name="Motel One Madrid", location="Madrid Latina", price_per_night=70.00, rating=3.9, amenities=["free wifi"]),
        HotelOption(name="Mandarin Oriental Tokyo", location="Tokyo Nihonbashi", price_per_night=500.00, rating=4.9, amenities=["free wifi", "spa", "pool"]),
        HotelOption(name="Capsule Inn Tokyo", location="Tokyo Shinjuku", price_per_night=35.00, rating=3.4, amenities=["free wifi"]),
        HotelOption(name="Shangri-La Sydney", location="Sydney Circular Quay", price_per_night=450.00, rating=4.8, amenities=["free wifi", "pool", "views"]),
        HotelOption(name="YHA Sydney", location="Sydney Kings Cross", price_per_night=45.00, rating=3.6, amenities=["free wifi", "kitchen"]),
    ]
    
    filtered_hotels = []
    for h in all_hotels:
        if destination.lower() in h.location.lower() and h.price_per_night <= budget:
            if "luxury" in preferences and h.rating >= 4.5 and "spa" in h.amenities:
                filtered_hotels.append(h)
            elif "budget" in preferences and h.price_per_night <= 100.00:
                filtered_hotels.append(h)
            elif "near beach" in preferences and "pool" in h.amenities:  # Simulate 'near beach' with pool
                filtered_hotels.append(h)
            elif not preferences:
                filtered_hotels.append(h)

    if not filtered_hotels:
        return ('hotel_options', [])  # Realistic: no matches

    # Return top 2 for realism
    filtered_hotels.sort(key=lambda x: -x.rating)  # Highest rated first
    return ('hotel_options', [h.model_dump() for h in filtered_hotels[:2]])

def search_activities_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for activities based on destination, dates, and interests. Adds realistic details like duration and best time.
    
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

    # Fake Data - More realistic with duration and best time, filtered by interests
    all_activities = [
        ActivityOption(name="Louvre Museum Visit", description="Explore world-famous art collections.", estimated_cost=20.00, duration="3 hours", best_time="morning"),
        ActivityOption(name="Eiffel Tower Climb", description="Ascend the iconic tower for views.", estimated_cost=25.00, duration="1 hour", best_time="evening"),
        ActivityOption(name="French Cooking Class", description="Learn to cook classic dishes.", estimated_cost=80.00, duration="4 hours", best_time="afternoon"),
        ActivityOption(name="Colosseum Guided Tour", description="Dive into ancient Roman history.", estimated_cost=50.00, duration="2 hours", best_time="morning"),
        ActivityOption(name="Vatican Museums Excursion", description="See Michelangelo's masterpieces.", estimated_cost=60.00, duration="3 hours", best_time="early morning"),
        ActivityOption(name="Italian Gelato Tasting", description="Sample authentic flavors.", estimated_cost=15.00, duration="1 hour", best_time="afternoon"),
        ActivityOption(name="Tower of London Tour", description="Discover royal history and jewels.", estimated_cost=30.00, duration="2 hours", best_time="morning"),
        ActivityOption(name="Thames River Cruise", description="See landmarks from the water.", estimated_cost=25.00, duration="1 hour", best_time="evening"),
        ActivityOption(name="Pub Crawl in London", description="Experience local nightlife.", estimated_cost=40.00, duration="4 hours", best_time="evening"),
        ActivityOption(name="Berlin Wall Bike Tour", description="Cycle along historical sites.", estimated_cost=35.00, duration="3 hours", best_time="daytime"),
        ActivityOption(name="Pergamon Museum Visit", description="View ancient artifacts.", estimated_cost=12.00, duration="2 hours", best_time="morning"),
        ActivityOption(name="Street Food Tour Berlin", description="Taste diverse cuisines.", estimated_cost=50.00, duration="2 hours", best_time="evening"),
        ActivityOption(name="Retiro Park Picnic", description="Relax in Madrid's green oasis.", estimated_cost=10.00, duration="2 hours", best_time="afternoon"),
        ActivityOption(name="Flamenco Show", description="Watch passionate Spanish dance.", estimated_cost=40.00, duration="1.5 hours", best_time="evening"),
        ActivityOption(name="Senso-ji Temple Visit", description="Explore Tokyo's oldest temple.", estimated_cost=0.00, duration="1 hour", best_time="morning"),
        ActivityOption(name="Mount Fuji Day Trip", description="View the iconic mountain.", estimated_cost=100.00, duration="8 hours", best_time="daytime"),
        ActivityOption(name="Sydney Harbour Cruise", description="Sail past Opera House and Bridge.", estimated_cost=50.00, duration="2 hours", best_time="evening"),
        ActivityOption(name="Blue Mountains Hike", description="Adventure through scenic trails.", estimated_cost=80.00, duration="6 hours", best_time="morning"),
    ]
    
    filtered_activities = []
    for a in all_activities:
        if destination.lower() in a.name.lower() or destination.lower() in a.description.lower():
            if "museums" in interests and "museum" in a.name.lower():
                filtered_activities.append(a)
            elif "adventure" in interests and ("climb" in a.name.lower() or "hike" in a.name.lower() or "bike" in a.name.lower()):
                filtered_activities.append(a)
            elif "food" in interests and ("food" in a.name.lower() or "tasting" in a.name.lower() or "cooking" in a.name.lower()):
                filtered_activities.append(a)
            elif "history" in interests and ("history" in a.description.lower() or "temple" in a.name.lower() or "tour" in a.name.lower()):
                filtered_activities.append(a)
            elif not interests:
                filtered_activities.append(a)

    if not filtered_activities:
        return ('activity_options', [])  # Realistic: no matches

    # Return up to 4 for a balanced trip
    return ('activity_options', [a.model_dump() for a in filtered_activities[:4]])

def calculate_total_cost_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Calculates the total estimated cost based on selected flights, hotels, and activities. Uses realistic night calculation from dates.
    
    Args:
        state: Current agent state (expects 'flight_options', 'hotel_options', 'activity_options', 'travel_dates' to be present).
        args: Dictionary with tool arguments (none needed, uses state).
            
    Returns:
        A tuple with ('total_estimated_cost', float).
    """
    from datetime import datetime
    total_cost = 0.0
    
    # Parse dates for num_nights
    dates = state.get('travel_dates', '')
    try:
        dep_date, ret_date = [datetime.strptime(d.strip(), '%Y-%m-%d') for d in dates.split(' to ')]
        num_nights = (ret_date - dep_date).days - 1  # Realistic: nights = days - 1
    except ValueError:
        num_nights = 6  # Fallback

    flights = state.get('flight_data', {}).get('options', [])
    for f in flights:
        total_cost += f['price']
        
    hotels = state.get('hotel_options', [])
    for h in hotels:
        total_cost += h['price_per_night'] * num_nights
        
    activities = state.get('activity_options', [])
    for a in activities:
        total_cost += a['estimated_cost']
        
    return ('total_estimated_cost', total_cost)

def generate_itinerary_summary_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Generates a summary of the travel itinerary based on selected flights, hotels, and activities. Includes more details for realism.
    
    Args:
        state: Current agent state (expects 'flight_options', 'hotel_options', 'activity_options' to be present).
        args: Dictionary with tool arguments (none needed, uses state).
            
    Returns:
        A tuple with ('itinerary_summary', str).
    """
    destination = state.get('destination', 'Unknown Destination')
    travel_dates = state.get('travel_dates', 'Unknown Dates')
    flight_data = state.get('flight_data', {'options': [], 'status': 'Not searched.'})
    flights = flight_data.get('options', [])
    flight_search_status = flight_data.get('status', 'Not searched.')

    summary = f"Detailed Travel Itinerary for {destination} ({travel_dates}):\n\n"
    
    summary += f"Flight Search Status: {flight_search_status}\n\n"

    if flights:
        summary += "Flight Details:\n"
        for f in flights:
            summary += f"  - {f['airline']} {f['flight_number']} from {f['origin']} to {f['destination']}: Departs {f['departure_time']}, Arrives {f['arrival_time']} (${f['price']:.2f}, {f['layovers']} layovers, Duration: {f['duration']})\n"
    
    if hotels:
        summary += "\nAccommodation:\n"
        for h in hotels:
            summary += f"  - {h['name']} in {h['location']}: ${h['price_per_night']:.2f}/night (Rating: {h['rating']}, Amenities: {', '.join(h['amenities'])})\n"
            
    if activities:
        summary += "\nPlanned Activities:\n"
        for a in activities:
            summary += f"  - {a['name']}: {a['description']} (Est. Cost: ${a['estimated_cost']:.2f}, Duration: {a['duration']}, Best Time: {a['best_time']})\n"
            
    summary += f"\nTotal Estimated Cost: ${total_cost:.2f} (Includes flights, hotels for duration, and activities. Note: Additional costs like meals/transport may apply.)\n"
    summary += "\nThis itinerary is designed for a balanced trip. Adjust based on weather or personal preferences."
    
    return ('itinerary_summary', summary)

# --- 3. Agent Configuration and Execution ---

if __name__ == "__main__":
    # Define the travel parameters
    travel_destination = "Rome"
    travel_origin = "London"
    travel_dates = "2025-09-10 to 2025-09-17"
    travel_budget = 10000.00 # Adjusted for realism with more costs
    hotel_preferences = "luxury"
    activity_interests = "history, food"

    # More realistic and detailed goal for the agent, simulating a user query
    agent_goal = (
        f"I need help planning a realistic trip from {travel_origin} to {travel_destination} for the dates {travel_dates}, "
        f"staying within a total budget of about ${travel_budget:.2f}. Please start by searching for affordable flights that fit the dates and budget. "
        f"If no flights are available, note that and suggest alternatives. Then, find {hotel_preferences} hotel options in {travel_destination} that match the dates and remaining budget. "
        f"After that, recommend activities focused on {activity_interests} that can be done during the trip. "
        f"Finally, calculate the total cost including all elements and provide a detailed, day-by-day style itinerary summary if possible, or a general overview."
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
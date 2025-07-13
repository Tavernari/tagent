import json
import sys
import os
from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

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

from datetime import datetime, timedelta  # Necessário para cálculos de data

def search_flights_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[List[Tuple[str, Any]]]:
    """
    Searches for flight options based on origin, destination, dates, and budget. 
    Simulates a real API by filtering flights on exact dates, with realistic variations like layovers and durations.
    
    This tool prioritizes arguments from 'args' (passed in the tool call), falling back to the agent 'state' if missing.
    If parameters are incomplete, it returns an empty result with an error status instead of failing silently.
    
    Args:
        state: Current agent state (dictionary with potential keys like 'origin', 'destination', 'travel_dates', 'budget').
        args: Dictionary with tool arguments (overrides state if provided):
            - origin (str): Departure city (e.g., 'London').
            - destination (str): Arrival city (e.g., 'Rome').
            - dates (str): Travel dates in format 'YYYY-MM-DD to YYYY-MM-DD' (e.g., '2025-09-10 to 2025-09-17').
            - budget (float): Maximum budget for flights (e.g., 2000.0).
            
    Returns:
        A list of tuples for state updates, e.g., [('flight_data', {'options': [...], 'status': 'Flights found.'})].
        If no flights match, returns an empty options list with a descriptive status.
    
    Example:
        Input args: {'origin': 'London', 'destination': 'Rome', 'dates': '2025-09-10 to 2025-09-17', 'budget': 500}
        Output: [('flight_data', {'options': [flight_dicts], 'status': 'Flights found successfully.'})]
    """
    # Priorize args, fallback para state
    origin = args.get('origin') or state.get('origin')
    destination = args.get('destination') or state.get('destination')
    dates = args.get('dates') or state.get('travel_dates')
    budget = args.get('budget') or state.get('budget')

    if not all([origin, destination, dates, budget]):
        return [('flight_data', {'options': [], 'status': 'Missing parameters for flight search (origin, destination, dates, or budget).'})]

    # Parse dates for realistic filtering
    try:
        dep_date, ret_date = [d.strip() for d in dates.split(' to ')]
        dep_dt = datetime.strptime(dep_date, '%Y-%m-%d')
        next_day_dt = dep_dt + timedelta(days=1)
        next_day_str = next_day_dt.strftime('%Y-%m-%d')
    except ValueError:
        return [('flight_data', {'options': [], 'status': 'Invalid date format (expected "YYYY-MM-DD to YYYY-MM-DD"). Unable to search flights.'})]

    # Fake data (simulated flights)
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
        FlightOption(airline="EasyJet", flight_number="EJ103", origin="London", destination="Rome", departure_time=f"2025-09-11 09:00", arrival_time=f"2025-09-11 11:30", price=150.00, layovers=0, duration="2h 30m"),
        FlightOption(airline="Ryanair", flight_number="RY104", origin="London", destination="Rome", departure_time=f"{dep_date} 06:00", arrival_time=f"{dep_date} 08:30", price=100.00, layovers=0, duration="2h 30m"),
        FlightOption(airline="United", flight_number="UA105", origin="New York", destination="Paris", departure_time=f"{dep_date} 18:00", arrival_time=f"{next_day_str} 08:00", price=600.00, layovers=0, duration="14h"),  # Overnight (corrigido)
    ]
    
    # Filter by route and date
    route_flights = [f for f in all_flights if f.origin.lower() == origin.lower() and f.destination.lower() == destination.lower() and dep_date in f.departure_time]
    
    if not route_flights:
        return [('flight_data', {'options': [], 'status': f"No flights found for {origin} to {destination} on {dep_date}."})]
    
    # Filter by budget
    affordable_flights = [f for f in route_flights if f.price <= budget]
    
    if not affordable_flights:
        return [('flight_data', {'options': [], 'status': f"No flights within budget ${budget:.2f} for {origin} to {destination} on {dep_date}."})]
    
    # Sort and return top 3
    affordable_flights.sort(key=lambda x: x.price)
    return [('flight_data', {'options': [f.model_dump() for f in affordable_flights[:3]], 'status': "Flights found successfully. Showing top 3 options."})]

def search_hotels_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for hotel options based on destination, dates, budget, and preferences. 
    Calculates number of nights from dates for realism and filters strictly by criteria.
    
    Prioritizes 'args' over 'state'. Returns empty list if no matches or missing params.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments (overrides state):
            - destination (str): City (e.g., 'Rome').
            - dates (str): 'YYYY-MM-DD to YYYY-MM-DD' (e.g., '2025-09-10 to 2025-09-17').
            - budget (float): Max per night (e.g., 500.0).
            - preferences (str): e.g., 'luxury' (filters by rating/amenities).
            
    Returns:
        Tuple: ('hotel_options', List[dict]) with hotel details.
    
    Example:
        Input args: {'destination': 'Rome', 'preferences': 'luxury'}
        Output: ('hotel_options', [{'name': 'Ritz Carlton Rome', ...}])
    """
    destination = args.get('destination') or state.get('destination')
    dates = args.get('dates') or state.get('travel_dates')
    budget = args.get('budget') or (state.get('budget', 0) / 7)
    preferences = args.get('preferences') or state.get('hotel_preferences', '').lower()

    if not all([destination, dates]):
        return ('hotel_options', [])

    # Parse dates (fallback if invalid)
    try:
        from datetime import datetime
        dep_date, ret_date = [datetime.strptime(d.strip(), '%Y-%m-%d') for d in dates.split(' to ')]
        num_nights = (ret_date - dep_date).days
        if num_nights <= 0:
            return ('hotel_options', [])
    except ValueError:
        num_nights = 6  # Fallback silencioso

    # Fake data and filter (rest same as original)
    all_hotels = [  # Fake hotel list, like in the original
        HotelOption(name="Four Seasons Paris", location="Paris Champs-Élysées", price_per_night=450.00, rating=4.8, amenities=["free wifi", "pool", "spa"]),
        # ... (add the rest of the list from original code here)
    ]
    
    filtered_hotels = []  # Filter logic same as original
    for h in all_hotels:
        if destination.lower() in h.location.lower() and h.price_per_night <= budget:
            if "luxury" in preferences and h.rating >= 4.5 and "spa" in h.amenities:
                filtered_hotels.append(h)
            # ... (resto da lógica de filtro)

    if not filtered_hotels:
        return ('hotel_options', [])

    filtered_hotels.sort(key=lambda x: -x.rating)
    return ('hotel_options', [h.model_dump() for h in filtered_hotels[:2]])

def search_activities_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for activities based on destination, dates, and interests. 
    Filters by interests with realistic details like duration and best time.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments:
            - destination (str): e.g., 'Rome'.
            - dates (str): e.g., '2025-09-10 to 2025-09-17'.
            - interests (str): e.g., 'history, food'.
            
    Returns:
        Tuple: ('activity_options', List[dict]).
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

from datetime import datetime  # Importe no topo do script se necessário

def calculate_total_cost_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[List[Tuple[str, Any]]]:
    """
    Calculates the total estimated cost of the trip based on selected flights, hotels, and activities from the agent state.
    - Flights: Sums prices from all flight options.
    - Hotels: Multiplies price_per_night by the number of nights (calculated from 'travel_dates' in state).
    - Activities: Sums estimated costs.
    
    This tool relies entirely on the 'state' (updated by previous tools) and does NOT use 'args' for inputs.
    If state data is missing or invalid, it uses fallbacks (e.g., 6 nights) and returns a status message for debugging.
    
    Args:
        state: Current agent state dictionary. Expected keys (populated by other tools):
            - 'flight_data' (dict): Contains 'options' (list of dicts with 'price' key, e.g., [{'price': 220.0}, ...]).
            - 'hotel_options' (list): List of dicts with 'price_per_night' (e.g., [{'price_per_night': 350.0}, ...]).
            - 'activity_options' (list): List of dicts with 'estimated_cost' (e.g., [{'estimated_cost': 50.0}, ...]).
            - 'travel_dates' (str): Dates in format 'YYYY-MM-DD to YYYY-MM-DD' (e.g., '2025-09-10 to 2025-09-17') to calculate nights.
              If missing/invalid, defaults to 6 nights.
        args: Dictionary with tool arguments (NOT USED in this tool; ignored to maintain compatibility with TAgent format).
              If you need to pass custom params (e.g., override budget), they are not supported here – use state updates instead.
            
    Returns:
        A list of tuples for state updates, e.g.:
            [('total_estimated_cost', 1500.0), ('cost_status', 'Calculated successfully.')]
        If data is missing, returns [('total_estimated_cost', 0.0), ('cost_status', 'Missing required data in state.')].
    
    Example:
        Input state: {
            'flight_data': {'options': [{'price': 220.0}]},
            'hotel_options': [{'price_per_night': 350.0}],
            'activity_options': [{'estimated_cost': 50.0}],
            'travel_dates': '2025-09-10 to 2025-09-17'  # 7 days = 6 nights
        }
        Output: [('total_estimated_cost', 220 + (350 * 6) + 50 = 2370.0), ('cost_status', 'Calculated successfully.')]
    
    Potential Issues:
        - If state lacks keys (e.g., no 'flight_data'), cost will be partial or 0.0. Ensure previous tools update state correctly.
        - Invalid 'travel_dates' uses fallback (6 nights) to avoid crashes, but check logs/state for accuracy.
        - Debug tip: Add print(state) here to inspect inputs during runs.
    """
    total_cost = 0.0
    cost_status = 'Calculated successfully.'
    
    # Parse dates for num_nights (fallback if invalid/missing)
    dates = state.get('travel_dates', '')
    try:
        if not dates:
            raise ValueError("No travel_dates in state.")
        dep_date, ret_date = [datetime.strptime(d.strip(), '%Y-%m-%d') for d in dates.split(' to ')]
        num_nights = max(0, (ret_date - dep_date).days - 1)  # Realistic: nights = days - 1, min 0
    except ValueError:
        num_nights = 6  # Fallback silencioso
        cost_status = 'Used fallback 6 nights due to invalid/missing travel_dates.'

    # Sum flights (safe access)
    flights = state.get('flight_data', {}).get('options', [])
    if not flights:
        cost_status = f'{cost_status} Warning: No flight data.'
    for f in flights:
        total_cost += f.get('price', 0.0)

    # Sum hotels * num_nights
    hotels = state.get('hotel_options', [])
    if not hotels:
        cost_status = f'{cost_status} Warning: No hotel data.'
    for h in hotels:
        total_cost += h.get('price_per_night', 0.0) * num_nights

    # Sum activities
    activities = state.get('activity_options', [])
    if not activities:
        cost_status = f'{cost_status} Warning: No activity data.'
    for a in activities:
        total_cost += a.get('estimated_cost', 0.0)

    # If total is 0 and status has warnings, mark as error
    if total_cost == 0.0 and 'Warning' in cost_status:
        cost_status = 'Missing required data in state (flights, hotels, activities, or dates). Total cost set to 0.'

    # Return list of tuples for multiple updates
    return [('total_estimated_cost', total_cost), ('cost_status', cost_status.strip())]

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
    hotels = state.get('hotel_options', [])  # Default vazio
    activities = state.get('activity_options', [])  # Default vazio
    total_cost = state.get('total_estimated_cost', 0.0)  # Default 0

    summary = f"Detailed Travel Itinerary for {destination} ({travel_dates}):\n\n"
    
    summary += f"Flight Search Status: {flight_search_status}\n\n"

    if flights:
        summary += "Flight Details:\n"
        for f in flights:
            summary += f"  - {f.get('airline', 'Unknown')} {f.get('flight_number', 'Unknown')} from {f.get('origin', 'Unknown')} to {f.get('destination', 'Unknown')}: Departs {f.get('departure_time', 'Unknown')}, Arrives {f.get('arrival_time', 'Unknown')} (${f.get('price', 0.0):.2f}, {f.get('layovers', 0)} layovers, Duration: {f.get('duration', 'Unknown')})\n"
    
    if hotels:
        summary += "\nAccommodation:\n"
        for h in hotels:
            summary += f"  - {h.get('name', 'Unknown')} in {h.get('location', 'Unknown')}: ${h.get('price_per_night', 0.0):.2f}/night (Rating: {h.get('rating', 0.0)}, Amenities: {', '.join(h.get('amenities', []))})\n"
            
    if activities:
        summary += "\nPlanned Activities:\n"
        for a in activities:
            summary += f"  - {a.get('name', 'Unknown')}: {a.get('description', 'Unknown')} (Est. Cost: ${a.get('estimated_cost', 0.0):.2f}, Duration: {a.get('duration', '')}, Best Time: {a.get('best_time', '')})\n"
            
    summary += f"\nTotal Estimated Cost: ${total_cost:.2f} (Includes flights, hotels for duration, and activities. Note: Additional costs like meals/transport may apply.)\n"
    summary += "\nThis itinerary is designed for a balanced trip. Adjust based on weather or personal preferences."
    
    if not (flights or hotels or activities):
        summary += "\nWarning: Limited data available - itinerary is incomplete."
    
    return ('itinerary_summary', summary)

# --- 3. Agent Configuration and Execution ---

if __name__ == "__main__":
    # Define the travel parameters
    travel_destination = "Rome"
    travel_origin = "London"
    travel_dates = "2025-09-10 to 2025-09-17"
    travel_budget = 10000.00 # Adjusted for realism with more costs
    hotel_preferences = "any"
    activity_interests = "food"

    # More realistic and detailed goal for the agent, simulating a user query
    # Mais realista e detalhado, com instruções para formato de resposta
    agent_goal = (
        f"I need help planning a realistic trip from {travel_origin} to {travel_destination} for the dates {travel_dates}, "
        f"staying within a total budget of about ${travel_budget:.2f}. Please start by searching for affordable flights that fit the dates and budget. "
        f"If no flights are available, note that and suggest alternatives. Then, find {hotel_preferences} hotel options in {travel_destination} that match the dates and remaining budget. "
        f"After that, recommend activities focused on {activity_interests} that can be done during the trip. "
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
        model="openrouter/google/gemini-2.5-pro", # Model the agent will use for decisions
        tools=agent_tools,
        output_format=TravelPlan,
        verbose=False
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
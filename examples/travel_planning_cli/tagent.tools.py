"""
Travel Planning Tools for TAgent CLI Example
Extracted tools from the original travel_planning_agent_example.py
"""

from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

# --- Data Models ---

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

# --- Tool Functions ---

def search_flights_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[List[Tuple[str, Any]]]:
    """
    Searches for flight options based on origin, destination, dates, and budget.
    Simulates a real API by filtering flights on exact dates, with realistic variations.
    """
    # Prioritize args, fallback to state
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

    # Mock flight data
    all_flights = [
        FlightOption(airline="Air France", flight_number="AF101", origin="New York", destination="Paris", departure_time=f"{dep_date} 08:00", arrival_time=f"{dep_date} 20:00", price=550.00, layovers=0, duration="12h"),
        FlightOption(airline="Delta", flight_number="DL205", origin="London", destination="Rome", departure_time=f"{dep_date} 10:30", arrival_time=f"{dep_date} 13:00", price=220.00, layovers=0, duration="2h 30m"),
        FlightOption(airline="Lufthansa", flight_number="LH300", origin="Berlin", destination="Madrid", departure_time=f"{dep_date} 14:00", arrival_time=f"{dep_date} 17:00", price=180.00, layovers=0, duration="3h"),
        FlightOption(airline="Qantas", flight_number="QF400", origin="Tokyo", destination="Sydney", departure_time=f"{dep_date} 09:00", arrival_time=f"{dep_date} 19:00", price=700.00, layovers=0, duration="10h"),
        FlightOption(airline="British Airways", flight_number="BA501", origin="New York", destination="London", departure_time=f"{dep_date} 11:00", arrival_time=f"{dep_date} 22:00", price=450.00, layovers=0, duration="11h"),
        FlightOption(airline="EasyJet", flight_number="EJ103", origin="London", destination="Rome", departure_time=f"2025-09-11 09:00", arrival_time=f"2025-09-11 11:30", price=150.00, layovers=0, duration="2h 30m"),
        FlightOption(airline="Ryanair", flight_number="RY104", origin="London", destination="Rome", departure_time=f"{dep_date} 06:00", arrival_time=f"{dep_date} 08:30", price=100.00, layovers=0, duration="2h 30m"),
        FlightOption(airline="United", flight_number="UA105", origin="New York", destination="Paris", departure_time=f"{dep_date} 18:00", arrival_time=f"{next_day_str} 08:00", price=600.00, layovers=0, duration="14h"),
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
    """
    destination = args.get('destination') or state.get('destination')
    dates = args.get('dates') or state.get('travel_dates')
    budget = args.get('budget') or (state.get('budget', 0) / 7)
    preferences = args.get('preferences') or state.get('hotel_preferences', '').lower()

    if not all([destination, dates]):
        return ('hotel_options', [])

    # Parse dates for nights calculation
    try:
        dep_date, ret_date = [datetime.strptime(d.strip(), '%Y-%m-%d') for d in dates.split(' to ')]
        num_nights = (ret_date - dep_date).days
        if num_nights <= 0:
            return ('hotel_options', [])
    except ValueError:
        num_nights = 6  # Fallback

    # Mock hotel data
    all_hotels = [
        HotelOption(name="Four Seasons Paris", location="Paris Champs-Élysées", price_per_night=450.00, rating=4.8, amenities=["free wifi", "pool", "spa"]),
        HotelOption(name="Ibis Paris", location="Paris Suburb", price_per_night=90.00, rating=3.5, amenities=["free wifi", "breakfast"]),
        HotelOption(name="Ritz Carlton Rome", location="Rome City Center", price_per_night=350.00, rating=4.7, amenities=["free wifi", "pool", "gym"]),
        HotelOption(name="Hostel Roma", location="Rome Termini", price_per_night=50.00, rating=3.0, amenities=["free wifi"]),
        HotelOption(name="The Savoy London", location="London Westminster", price_per_night=400.00, rating=4.9, amenities=["free wifi", "spa", "restaurant"]),
        HotelOption(name="Premier Inn London", location="London East End", price_per_night=80.00, rating=3.8, amenities=["free wifi", "breakfast"]),
        HotelOption(name="Hotel Adlon Berlin", location="Berlin Mitte", price_per_night=300.00, rating=4.6, amenities=["free wifi", "pool", "bar"]),
        HotelOption(name="A&O Hostel Berlin", location="Berlin Kreuzberg", price_per_night=40.00, rating=3.2, amenities=["free wifi"]),
    ]
    
    # Filter by destination and budget
    filtered_hotels = []
    for h in all_hotels:
        if destination.lower() in h.location.lower() and h.price_per_night <= budget:
            if "luxury" in preferences and h.rating >= 4.5 and "spa" in h.amenities:
                filtered_hotels.append(h)
            elif "budget" in preferences and h.price_per_night <= 100.00:
                filtered_hotels.append(h)
            elif not preferences or "any" in preferences:
                filtered_hotels.append(h)

    if not filtered_hotels:
        return ('hotel_options', [])

    # Sort by rating and return top 2
    filtered_hotels.sort(key=lambda x: -x.rating)
    return ('hotel_options', [h.model_dump() for h in filtered_hotels[:2]])

def search_activities_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for activities based on destination and interests.
    """
    destination = args.get('destination') or state.get('destination')
    dates = args.get('dates') or state.get('travel_dates')
    interests = args.get('interests') or state.get('activity_interests', '').lower()

    if not all([destination, dates]):
        return None

    # Mock activity data
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
    ]
    
    # Filter by destination and interests
    filtered_activities = []
    for a in all_activities:
        if destination.lower() in a.name.lower() or destination.lower() in a.description.lower():
            if "museums" in interests and "museum" in a.name.lower():
                filtered_activities.append(a)
            elif "adventure" in interests and ("climb" in a.name.lower() or "bike" in a.name.lower()):
                filtered_activities.append(a)
            elif "food" in interests and ("food" in a.name.lower() or "tasting" in a.name.lower() or "cooking" in a.name.lower() or "gelato" in a.name.lower()):
                filtered_activities.append(a)
            elif "history" in interests and ("history" in a.description.lower() or "museum" in a.name.lower() or "tour" in a.name.lower()):
                filtered_activities.append(a)
            elif not interests:
                filtered_activities.append(a)

    if not filtered_activities:
        return ('activity_options', [])

    # Return up to 4 activities
    return ('activity_options', [a.model_dump() for a in filtered_activities[:4]])

def calculate_total_cost_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[List[Tuple[str, Any]]]:
    """
    Calculates the total estimated cost of the trip based on flights, hotels, and activities.
    """
    total_cost = 0.0
    cost_status = 'Calculated successfully.'
    
    # Parse dates for nights calculation
    dates = state.get('travel_dates', '')
    try:
        if not dates:
            raise ValueError("No travel_dates in state.")
        dep_date, ret_date = [datetime.strptime(d.strip(), '%Y-%m-%d') for d in dates.split(' to ')]
        num_nights = max(0, (ret_date - dep_date).days - 1)
    except ValueError:
        num_nights = 6  # Fallback
        cost_status = 'Used fallback 6 nights due to invalid/missing travel_dates.'

    # Sum flights
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

    # Check if we have sufficient data
    if total_cost == 0.0 and 'Warning' in cost_status:
        cost_status = 'Missing required data in state (flights, hotels, activities, or dates). Total cost set to 0.'

    return [('total_estimated_cost', total_cost), ('cost_status', cost_status.strip())]

def generate_itinerary_summary_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Generates a detailed summary of the travel itinerary.
    """
    destination = state.get('destination', 'Unknown Destination')
    travel_dates = state.get('travel_dates', 'Unknown Dates')
    flight_data = state.get('flight_data', {'options': [], 'status': 'Not searched.'})
    flights = flight_data.get('options', [])
    flight_search_status = flight_data.get('status', 'Not searched.')
    hotels = state.get('hotel_options', [])
    activities = state.get('activity_options', [])
    total_cost = state.get('total_estimated_cost', 0.0)

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
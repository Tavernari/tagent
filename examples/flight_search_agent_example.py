import json
import sys
import os
from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import random

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tagent.agent import run_agent

# --- 1. Pydantic Model Definition for the Final Output ---
# This model ensures the agent's output is a structured and validated JSON.

class FlightSearchResult(BaseModel):
    cheapest_option: Dict[str, Any] = Field(..., description="The cheapest flight option (direct or with connections).")
    cheapest_direct: Dict[str, Any] = Field(..., description="The cheapest direct flight.")
    cheapest_connection: Optional[Dict[str, Any]] = Field(None, description="The cheapest connection option if found.")
    all_flights: List[Dict[str, Any]] = Field(..., description="A list of all flights found during the search.")
    connection_routes: List[Dict[str, Any]] = Field(default=[], description="A list of connection routes found.")
    savings_analysis: Dict[str, Any] = Field(..., description="Analysis of potential savings with connections.")
    total_flights_searched: int = Field(..., description="Total number of flights searched.")
    summary: str = Field(..., description="An AI-generated summary of the flight search results.")

# --- 2. Fake Tool Definitions ---
# Each function simulates flight search APIs and returns flight data.

def search_flights_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for flights between two cities on a specific date.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments.
            - origin (str): Origin city code (e.g., "NYC")
            - destination (str): Destination city code (e.g., "LAX")
            - date (str): Flight date in YYYY-MM-DD format
            
    Returns:
        A tuple with ('flights_found', flights) where flights is a list of flight options.
        
    Example:
        args = {"origin": "NYC", "destination": "LAX", "date": "2024-08-15"}
    """
    origin = args.get('origin')
    destination = args.get('destination')
    date = args.get('date')
    
    if not all([origin, destination, date]):
        return None

    # Simulate API connection delay
    print(f"🔍 Searching flights from {origin} to {destination} on {date}...")
    
    # Generate fake flight data with some randomness
    airlines = ["Delta", "American", "United", "Southwest", "JetBlue", "Alaska"]
    base_prices = [299, 345, 412, 389, 567, 234, 678, 445, 523, 367]
    
    flights = []
    num_flights = random.randint(3, 8)
    
    for i in range(num_flights):
        flight_number = f"{random.choice(airlines)[:2].upper()}{random.randint(100, 9999)}"
        base_price = random.choice(base_prices)
        # Add some price variation
        price = base_price + random.randint(-50, 150)
        
        # Generate departure and arrival times
        dept_hour = random.randint(6, 22)
        dept_minute = random.choice([0, 15, 30, 45])
        flight_duration = random.randint(180, 420)  # 3-7 hours in minutes
        
        arrival_time = dept_hour * 60 + dept_minute + flight_duration
        arrival_hour = (arrival_time // 60) % 24
        arrival_minute = arrival_time % 60
        
        flight = {
            "flight_number": flight_number,
            "airline": random.choice(airlines),
            "origin": origin,
            "destination": destination,
            "date": date,
            "departure_time": f"{dept_hour:02d}:{dept_minute:02d}",
            "arrival_time": f"{arrival_hour:02d}:{arrival_minute:02d}",
            "price": price,
            "duration_minutes": flight_duration,
            "stops": random.choice([0, 0, 0, 1, 1, 2])  # Mostly direct flights
        }
        flights.append(flight)
    
    # Update state with found flights
    current_flights = state.get('flights_found', [])
    current_flights.extend(flights)
    
    print(f"✅ Found {len(flights)} flights from {origin} to {destination}")
    return ('flights_found', current_flights)

def compare_prices_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Compares prices of all found flights and identifies the cheapest option.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments (none needed for this tool).
            
    Returns:
        A tuple with ('price_comparison', comparison_data) containing cheapest flight and statistics.
    """
    flights = state.get('flights_found', [])
    
    if not flights:
        return ('price_comparison', {"error": "No flights found to compare"})
    
    print(f"💰 Comparing prices for {len(flights)} flights...")
    
    # Find cheapest flight
    cheapest_flight = min(flights, key=lambda f: f['price'])
    most_expensive = max(flights, key=lambda f: f['price'])
    average_price = sum(f['price'] for f in flights) / len(flights)
    
    # Group by airline for analysis
    airline_prices = {}
    for flight in flights:
        airline = flight['airline']
        if airline not in airline_prices:
            airline_prices[airline] = []
        airline_prices[airline].append(flight['price'])
    
    airline_averages = {airline: sum(prices)/len(prices) 
                       for airline, prices in airline_prices.items()}
    
    comparison_data = {
        "cheapest_flight": cheapest_flight,
        "most_expensive_flight": most_expensive,
        "average_price": round(average_price, 2),
        "price_range": most_expensive['price'] - cheapest_flight['price'],
        "airline_averages": airline_averages,
        "total_flights_compared": len(flights)
    }
    
    print(f"🏆 Cheapest direct flight: {cheapest_flight['flight_number']} - ${cheapest_flight['price']}")
    return ('price_comparison', comparison_data)

def search_connections_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for flight connections through intermediate cities to find cheaper alternatives.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments.
            - origin (str): Origin city code (e.g., "NYC")
            - destination (str): Destination city code (e.g., "LAX")
            - date (str): Flight date in YYYY-MM-DD format
            
    Returns:
        A tuple with ('connection_routes', routes) where routes is a list of connection options.
        
    Example:
        args = {"origin": "NYC", "destination": "LAX", "date": "2024-08-15"}
    """
    origin = args.get('origin')
    destination = args.get('destination')
    date = args.get('date')
    
    if not all([origin, destination, date]):
        return None

    print(f"🔗 Searching connection routes from {origin} to {destination}...")
    
    # Common hub cities for connections
    hub_cities = ["ATL", "DFW", "ORD", "DEN", "PHX", "LAS", "IAH", "SEA", "MIA"]
    
    # Remove origin and destination from potential hubs
    available_hubs = [hub for hub in hub_cities if hub not in [origin, destination]]
    
    connection_routes = []
    
    # Generate 2-4 connection options
    num_connections = random.randint(2, 4)
    selected_hubs = random.sample(available_hubs, min(num_connections, len(available_hubs)))
    
    for hub in selected_hubs:
        # Generate first leg: origin to hub
        first_leg_price = random.randint(150, 400)
        first_leg_duration = random.randint(90, 300)
        
        # Departure time for first leg
        first_dept_hour = random.randint(6, 20)
        first_dept_minute = random.choice([0, 15, 30, 45])
        
        # Arrival time for first leg
        first_arrival_minutes = first_dept_hour * 60 + first_dept_minute + first_leg_duration
        first_arrival_hour = (first_arrival_minutes // 60) % 24
        first_arrival_minute = first_arrival_minutes % 60
        
        # Layover time (45 minutes to 4 hours)
        layover_minutes = random.randint(45, 240)
        
        # Second leg departure time
        second_dept_minutes = first_arrival_minutes + layover_minutes
        second_dept_hour = (second_dept_minutes // 60) % 24
        second_dept_minute = second_dept_minutes % 60
        
        # Generate second leg: hub to destination
        second_leg_price = random.randint(120, 350)
        second_leg_duration = random.randint(90, 300)
        
        # Second leg arrival
        second_arrival_minutes = second_dept_minutes + second_leg_duration
        second_arrival_hour = (second_arrival_minutes // 60) % 24
        second_arrival_minute = second_arrival_minutes % 60
        
        # Airlines for each leg
        airlines = ["Delta", "American", "United", "Southwest", "JetBlue", "Alaska"]
        first_airline = random.choice(airlines)
        second_airline = random.choice(airlines)
        
        # Total price with potential connection discount
        base_total = first_leg_price + second_leg_price
        # Sometimes connections are cheaper due to airline partnerships
        connection_discount = random.randint(-50, 30)  # Can be discount or premium
        total_price = base_total + connection_discount
        
        total_duration = first_leg_duration + layover_minutes + second_leg_duration
        
        connection_route = {
            "route_type": "connection",
            "origin": origin,
            "destination": destination,
            "hub_city": hub,
            "total_price": total_price,
            "total_duration_minutes": total_duration,
            "layover_minutes": layover_minutes,
            "legs": [
                {
                    "flight_number": f"{first_airline[:2].upper()}{random.randint(100, 9999)}",
                    "airline": first_airline,
                    "origin": origin,
                    "destination": hub,
                    "departure_time": f"{first_dept_hour:02d}:{first_dept_minute:02d}",
                    "arrival_time": f"{first_arrival_hour:02d}:{first_arrival_minute:02d}",
                    "duration_minutes": first_leg_duration,
                    "price": first_leg_price
                },
                {
                    "flight_number": f"{second_airline[:2].upper()}{random.randint(100, 9999)}",
                    "airline": second_airline,
                    "origin": hub,
                    "destination": destination,
                    "departure_time": f"{second_dept_hour:02d}:{second_dept_minute:02d}",
                    "arrival_time": f"{second_arrival_hour:02d}:{second_arrival_minute:02d}",
                    "duration_minutes": second_leg_duration,
                    "price": second_leg_price
                }
            ]
        }
        
        connection_routes.append(connection_route)
    
    # Sort by total price
    connection_routes.sort(key=lambda x: x['total_price'])
    
    print(f"✅ Found {len(connection_routes)} connection routes")
    for i, route in enumerate(connection_routes[:3], 1):  # Show top 3
        print(f"   {i}. {route['origin']} → {route['hub_city']} → {route['destination']}: ${route['total_price']} ({route['total_duration_minutes']//60}h{route['total_duration_minutes']%60}m)")
    
    return ('connection_routes', connection_routes)

def check_availability_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Simulates checking seat availability for the cheapest flight.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments.
            - flight_number (str): The flight number to check availability for.
            
    Returns:
        A tuple with ('availability_check', availability_info) containing seat availability data.
        
    Example:
        args = {"flight_number": "DL1234"}
    """
    flight_number = args.get('flight_number')
    
    if not flight_number:
        return None
    
    print(f"🪑 Checking seat availability for flight {flight_number}...")
    
    # Simulate availability check
    availability = {
        "flight_number": flight_number,
        "economy_seats": random.randint(5, 50),
        "business_seats": random.randint(0, 8),
        "first_class_seats": random.randint(0, 4),
        "total_available": 0,
        "booking_status": "available"
    }
    
    availability["total_available"] = (availability["economy_seats"] + 
                                     availability["business_seats"] + 
                                     availability["first_class_seats"])
    
    if availability["total_available"] < 5:
        availability["booking_status"] = "limited"
    elif availability["total_available"] == 0:
        availability["booking_status"] = "sold_out"
    
    print(f"✅ Availability check complete: {availability['total_available']} seats available")
    return ('availability_check', availability)

def analyze_savings_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Analyzes potential savings between direct flights and connections.
    
    Args:
        state: Current agent state.
        args: Dictionary with tool arguments (none needed for this tool).
            
    Returns:
        A tuple with ('savings_analysis', analysis_data) containing savings comparison.
    """
    direct_flights = state.get('flights_found', [])
    connection_routes = state.get('connection_routes', [])
    
    if not direct_flights:
        return ('savings_analysis', {"error": "No direct flights to compare"})
    
    print(f"💡 Analyzing savings between direct flights and connections...")
    
    # Find cheapest direct flight
    cheapest_direct = min(direct_flights, key=lambda f: f['price'])
    
    analysis = {
        "cheapest_direct_price": cheapest_direct['price'],
        "cheapest_direct_flight": cheapest_direct,
        "connection_options": [],
        "best_savings": 0,
        "best_connection": None,
        "recommendation": "direct"
    }
    
    if connection_routes:
        # Find cheapest connection
        cheapest_connection = min(connection_routes, key=lambda r: r['total_price'])
        
        savings = cheapest_direct['price'] - cheapest_connection['total_price']
        time_difference = cheapest_connection['total_duration_minutes'] - cheapest_direct['duration_minutes']
        
        analysis.update({
            "cheapest_connection_price": cheapest_connection['total_price'],
            "cheapest_connection_route": cheapest_connection,
            "best_savings": savings,
            "time_difference_minutes": time_difference,
            "best_connection": cheapest_connection
        })
        
        # Determine recommendation
        if savings > 100:  # Save more than $100
            if time_difference < 240:  # Less than 4 hours extra
                analysis["recommendation"] = "connection"
                analysis["recommendation_reason"] = f"Save ${savings:.0f} with only {time_difference//60}h{time_difference%60}m extra travel time"
            else:
                analysis["recommendation"] = "direct"
                analysis["recommendation_reason"] = f"Despite ${savings:.0f} savings, connection adds {time_difference//60}h{time_difference%60}m extra time"
        elif savings > 50:  # Moderate savings
            if time_difference < 120:  # Less than 2 hours extra
                analysis["recommendation"] = "connection"
                analysis["recommendation_reason"] = f"Moderate savings of ${savings:.0f} with acceptable {time_difference//60}h{time_difference%60}m extra time"
            else:
                analysis["recommendation"] = "direct"
                analysis["recommendation_reason"] = f"Only ${savings:.0f} savings not worth {time_difference//60}h{time_difference%60}m extra time"
        else:
            analysis["recommendation"] = "direct"
            analysis["recommendation_reason"] = f"Minimal savings of ${savings:.0f} not worth complexity of connections"
        
        # Show comparison
        if savings > 0:
            print(f"💰 Potential savings: ${savings:.0f} with connections")
            print(f"⏱️  Extra travel time: {time_difference//60}h{time_difference%60}m")
            print(f"🎯 Recommendation: {analysis['recommendation']} - {analysis['recommendation_reason']}")
        else:
            print(f"💸 Connections are ${abs(savings):.0f} more expensive than direct flights")
            analysis["recommendation"] = "direct"
            analysis["recommendation_reason"] = f"Direct flights are ${abs(savings):.0f} cheaper than connections"
    else:
        analysis["recommendation_reason"] = "No connection routes found, direct flight is only option"
        print("ℹ️  No connection routes found for comparison")
    
    return ('savings_analysis', analysis)

# --- 3. Agent Configuration and Execution ---

if __name__ == "__main__":
    # The clear goal for the agent
    agent_goal = """Find the absolute cheapest way to fly from New York (NYC) to Los Angeles (LAX) for August 15th, 2024. 
    Search for both direct flights AND connection routes through other cities. Compare all options including 
    potential savings from connections vs direct flights. Analyze if any connection routes offer significant 
    savings compared to direct flights. Check seat availability for the best option found. 
    Provide a comprehensive summary with recommendations."""

    # Dictionary registering the available tools
    agent_tools = {
        "search_flights": search_flights_tool,
        "search_connections": search_connections_tool,
        "compare_prices": compare_prices_tool,
        "analyze_savings": analyze_savings_tool,
        "check_availability": check_availability_tool,
    }

    print("--- Starting Flight Search Agent ---")
    
    # Execute the agent loop
    # The agent will use an LLM to decide which tool to call at each step.
    # The `output_format` ensures the final output is a `FlightSearchResult` object.
    final_output = run_agent(
        goal=agent_goal,
        model="openrouter/google/gemma-3-27b-it",  # Model the agent will use for decisions
        tools=agent_tools,
        output_format=FlightSearchResult,
        verbose=False,
    )

    print("--- Final Agent Result ---")
    if final_output:
        # Check for chat history in the result
        if isinstance(final_output, dict) and 'chat_summary' in final_output:
            # print("\n" + final_output['chat_summary'])
            
            # Check the execution status
            status = final_output.get('status', 'unknown')
            # print(f"\n--- STATUS: {status.upper().replace('_', ' ')} ---")
            
            if final_output['result']:
                # print("\n--- STRUCTURED RESULT ---")
                # The Pydantic model can be easily converted to a JSON
                json_output = final_output['result'].model_dump_json(indent=4)
                print(json_output)
            elif final_output.get('raw_data'):
                # print("\n--- COLLECTED DATA (UNFORMATTED) ---")
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
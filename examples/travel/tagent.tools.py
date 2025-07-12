"""
Travel Tools for TAgent
Contains tools for flight search, hotel booking, and activity recommendations.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json

def search_flights_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[List[Tuple[str, Any]]]:
    """
    Searches for flight options based on origin, destination, dates, and budget.
    
    Args:
        state: Current agent state
        args: Dictionary with:
            - origin (str): Departure city
            - destination (str): Arrival city  
            - dates (str): Travel dates (YYYY-MM-DD to YYYY-MM-DD)
            - budget (float): Maximum budget for flights
            
    Returns:
        List of tuples for state updates
    """
    origin = args.get('origin') or state.get('origin')
    destination = args.get('destination') or state.get('destination')
    dates = args.get('dates') or state.get('travel_dates')
    budget = args.get('budget') or state.get('budget')

    if not all([origin, destination, dates, budget]):
        return [('flight_search_result', {
            'success': False,
            'error': 'Missing required parameters (origin, destination, dates, budget)'
        })]

    try:
        dep_date, ret_date = [d.strip() for d in dates.split(' to ')]
        datetime.strptime(dep_date, '%Y-%m-%d')
        datetime.strptime(ret_date, '%Y-%m-%d')
    except ValueError:
        return [('flight_search_result', {
            'success': False,
            'error': 'Invalid date format. Use YYYY-MM-DD to YYYY-MM-DD'
        })]

    # Mock flight data
    flights = [
        {
            'airline': 'Air France',
            'flight_number': 'AF101',
            'origin': origin,
            'destination': destination,
            'departure': f'{dep_date} 08:00',
            'arrival': f'{dep_date} 20:00',
            'price': 550.00,
            'duration': '12h'
        },
        {
            'airline': 'Delta',
            'flight_number': 'DL205', 
            'origin': origin,
            'destination': destination,
            'departure': f'{dep_date} 10:30',
            'arrival': f'{dep_date} 13:00',
            'price': 220.00,
            'duration': '2h 30m'
        }
    ]
    
    # Filter by budget
    affordable_flights = [f for f in flights if f['price'] <= budget]
    
    return [('flight_search_result', {
        'success': True,
        'flights': affordable_flights,
        'search_params': {'origin': origin, 'destination': destination, 'dates': dates, 'budget': budget}
    })]

def search_hotels_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Searches for hotel options based on destination and preferences.
    
    Args:
        state: Current agent state
        args: Dictionary with:
            - destination (str): City to search
            - budget_per_night (float): Maximum per night
            - preferences (str): Hotel preferences (luxury, budget, etc.)
            
    Returns:
        Tuple with hotel search results
    """
    destination = args.get('destination') or state.get('destination')
    budget_per_night = args.get('budget_per_night', 200)
    preferences = args.get('preferences', '').lower()

    if not destination:
        return ('hotel_search_result', {
            'success': False,
            'error': 'Destination is required'
        })

    # Mock hotel data
    hotels = [
        {
            'name': 'Luxury Palace Hotel',
            'location': f'{destination} City Center',
            'price_per_night': 450.00,
            'rating': 4.8,
            'amenities': ['spa', 'pool', 'restaurant', 'wifi']
        },
        {
            'name': 'Budget Inn',
            'location': f'{destination} Downtown',
            'price_per_night': 80.00,
            'rating': 3.5,
            'amenities': ['wifi', 'breakfast']
        },
        {
            'name': 'Boutique Hotel',
            'location': f'{destination} Historic District',
            'price_per_night': 180.00,
            'rating': 4.2,
            'amenities': ['wifi', 'restaurant', 'gym']
        }
    ]
    
    # Filter by budget
    affordable_hotels = [h for h in hotels if h['price_per_night'] <= budget_per_night]
    
    # Filter by preferences
    if 'luxury' in preferences:
        affordable_hotels = [h for h in affordable_hotels if h['rating'] >= 4.5]
    elif 'budget' in preferences:
        affordable_hotels = [h for h in affordable_hotels if h['price_per_night'] <= 100]
    
    return ('hotel_search_result', {
        'success': True,
        'hotels': affordable_hotels[:3],  # Top 3 results
        'search_params': {'destination': destination, 'budget_per_night': budget_per_night, 'preferences': preferences}
    })

def get_activity_recommendations_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Gets activity recommendations for a destination.
    
    Args:
        state: Current agent state
        args: Dictionary with:
            - destination (str): City to get activities for
            - interests (str): User interests (history, food, adventure, etc.)
            - days (int): Number of days for activities
            
    Returns:
        Tuple with activity recommendations
    """
    destination = args.get('destination') or state.get('destination')
    interests = args.get('interests', '').lower()
    days = args.get('days', 3)

    if not destination:
        return ('activity_recommendations', {
            'success': False,
            'error': 'Destination is required'
        })

    # Mock activity data
    all_activities = [
        {
            'name': 'Historical Museum Tour',
            'category': 'history',
            'duration': '3 hours',
            'price': 25.00,
            'rating': 4.5,
            'description': 'Explore the rich history of the city'
        },
        {
            'name': 'Food Walking Tour', 
            'category': 'food',
            'duration': '4 hours',
            'price': 65.00,
            'rating': 4.8,
            'description': 'Taste local cuisine and specialties'
        },
        {
            'name': 'Adventure Park',
            'category': 'adventure',
            'duration': '6 hours', 
            'price': 45.00,
            'rating': 4.3,
            'description': 'Outdoor activities and thrills'
        },
        {
            'name': 'Art Gallery Visit',
            'category': 'culture',
            'duration': '2 hours',
            'price': 15.00,
            'rating': 4.1,
            'description': 'Contemporary and classical art collections'
        }
    ]
    
    # Filter by interests
    if interests:
        filtered_activities = [a for a in all_activities if interests in a['category'] or interests in a['name'].lower()]
    else:
        filtered_activities = all_activities
    
    # Limit by number of days
    recommended_activities = filtered_activities[:days]
    
    return ('activity_recommendations', {
        'success': True,
        'activities': recommended_activities,
        'total_estimated_cost': sum(a['price'] for a in recommended_activities),
        'search_params': {'destination': destination, 'interests': interests, 'days': days}
    })

def calculate_trip_budget_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Calculates total trip budget based on flights, hotels, and activities.
    
    Args:
        state: Current agent state (should contain search results)
        args: Dictionary with:
            - nights (int): Number of nights for hotel calculation
            
    Returns:
        Tuple with budget calculation
    """
    nights = args.get('nights', 7)
    
    total_cost = 0.0
    breakdown = {}
    
    # Flight costs
    flight_result = state.get('flight_search_result', {})
    if flight_result.get('success') and flight_result.get('flights'):
        flight_cost = min(f['price'] for f in flight_result['flights'])
        total_cost += flight_cost
        breakdown['flights'] = flight_cost
    
    # Hotel costs
    hotel_result = state.get('hotel_search_result', {})
    if hotel_result.get('success') and hotel_result.get('hotels'):
        hotel_per_night = min(h['price_per_night'] for h in hotel_result['hotels'])
        hotel_cost = hotel_per_night * nights
        total_cost += hotel_cost
        breakdown['hotels'] = hotel_cost
        breakdown['hotel_per_night'] = hotel_per_night
    
    # Activity costs
    activity_result = state.get('activity_recommendations', {})
    if activity_result.get('success'):
        activity_cost = activity_result.get('total_estimated_cost', 0)
        total_cost += activity_cost
        breakdown['activities'] = activity_cost
    
    return ('trip_budget_calculation', {
        'total_cost': total_cost,
        'breakdown': breakdown,
        'nights': nights,
        'currency': 'USD'
    })
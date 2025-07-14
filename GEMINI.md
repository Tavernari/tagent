# TAgent Technical Documentation

## Development Guidelines

Activate .venv first of all.

Always write code and comments in technical English. This ensures:
- International collaboration compatibility
- Clear technical communication
- Industry standard practices
- Better code maintainability

## Code Style Standards

### Language Requirements
- **Primary Language**: English for all code, comments, docstrings, and variable names
- **Documentation**: Technical English with precise terminology
- **Comments**: Explain the "why" not the "what" in English
- **Naming**: Use descriptive English names for variables, functions, and classes

### Code Quality
- Follow PEP 8 for Python code style
- Use type hints for all function signatures
- Write comprehensive docstrings for public APIs
- Maintain consistent formatting with Black
- Ensure lint-free code with flake8

### Architecture Principles
- Modular design with clear separation of concerns
- Redux-inspired state management pattern
- State machine-controlled action flow to prevent infinite loops
- Structured outputs over function calls for LLM compatibility
- Type-safe operations with Pydantic validation
- Production-ready error handling and logging

### Example Code Standards

```python
def search_flights_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[List[Tuple[str, Any]]]:
    """
    Search for flight options based on origin, destination, dates, and budget.
    
    This tool simulates a real flight API by filtering mock data based on exact 
    dates and applying realistic business logic like budget constraints and 
    availability windows.
    
    Args:
        state: Current agent state dictionary containing contextual information
        args: Tool arguments with search parameters:
            - origin (str): Departure city (e.g., 'London')
            - destination (str): Arrival city (e.g., 'Rome') 
            - dates (str): Travel dates in 'YYYY-MM-DD to YYYY-MM-DD' format
            - budget (float): Maximum budget for flights in USD
            
    Returns:
        List of state update tuples, typically [('flight_data', search_results)]
        Returns empty results with error status if parameters are invalid
        
    Example:
        >>> search_flights_tool(state, {
        ...     'origin': 'London', 
        ...     'destination': 'Rome',
        ...     'dates': '2025-09-10 to 2025-09-17', 
        ...     'budget': 500.0
        ... })
        [('flight_data', {'options': [...], 'status': 'Flights found successfully.'})]
    """
    # Implementation follows...
```

### Documentation Requirements
- All public functions must have comprehensive docstrings
- Include parameter types, return values, and usage examples
- Document error conditions and edge cases
- Provide architectural context where relevant
- Use technical English throughout

## State Machine Architecture (v0.5.0)

### Mandatory Action Flow
The agent follows a strict state machine to prevent infinite loops:

```
INITIAL → PLAN (mandatory)
PLAN → EXECUTE (mandatory) 
EXECUTE → PLAN | EXECUTE | SUMMARIZE (AI chooses)
SUMMARIZE → EVALUATE (mandatory)
EVALUATE → PLAN (mandatory, returns to cycle)
```

### Key Features
- **Automatic Path Following**: Single valid actions execute automatically
- **AI Decision Points**: Multiple options allow AI to choose the best path
- **Loop Prevention**: State transitions prevent SUMMARIZE→SUMMARIZE and EVALUATE→EVALUATE
- **Auto-Execution**: SUMMARIZE automatically triggers EVALUATE, failed EVALUATE triggers PLAN
- **Controlled Flow**: LLM choices are constrained to valid state transitions only
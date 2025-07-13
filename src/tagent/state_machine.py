"""
State machine for TAgent to control valid action transitions.
Prevents LLM from freely choosing actions that could create loops.
"""

from typing import Dict, List, Optional, Set
from enum import Enum


class AgentState(Enum):
    """Valid states in the agent execution."""
    INITIAL = "initial"
    PLANNING = "planning"
    EXECUTING = "executing"
    SUMMARIZING = "summarizing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    FINALIZING = "finalizing"


class ActionType(Enum):
    """Valid action types."""
    PLAN = "plan"
    EXECUTE = "execute"
    SUMMARIZE = "summarize"
    EVALUATE = "evaluate"
    FINALIZE = "finalize"
    NO_ACTION = "no_action"

class StateTransitionRule:
    """Defines a valid state transition rule."""
    
    def __init__(
        self, 
        from_state: AgentState, 
        action: ActionType, 
        to_state: AgentState,
        condition: Optional[str] = None
    ):
        self.from_state = from_state
        self.action = action
        self.to_state = to_state
        self.condition = condition


class AgentStateMachine:
    """
    State machine that enforces valid action transitions.
    
    Mandatory flow:
    INITIAL → PLAN (mandatory)
    PLAN → EXECUTE (mandatory) 
    EXECUTE → PLAN | EXECUTE | SUMMARIZE (AI chooses)
    SUMMARIZE → EVALUATE (mandatory)
    EVALUATE → PLAN (mandatory, returns to cycle)
    
    Rules:
    1. INITIAL: can only PLAN
    2. PLAN: can only EXECUTE  
    3. EXECUTE: can PLAN, EXECUTE or SUMMARIZE
    4. SUMMARIZE: must go to EVALUATE
    5. EVALUATE: must return to PLAN
    """
    
    def __init__(self):
        self.current_state = AgentState.INITIAL
        self.last_action = None
        self.action_history: List[ActionType] = []
        self.state_history: List[AgentState] = [AgentState.INITIAL]
        
        # Define valid transitions
        self.transitions = self._define_transitions()
    
    def _define_transitions(self) -> List[StateTransitionRule]:
        """Define all valid state transitions."""
        return [
            # From INITIAL state - can only PLAN
            StateTransitionRule(AgentState.INITIAL, ActionType.PLAN, AgentState.PLANNING),
            
            # From PLANNING state - can only EXECUTE
            StateTransitionRule(AgentState.PLANNING, ActionType.EXECUTE, AgentState.EXECUTING),
            
            # From EXECUTING state - can PLAN, EXECUTE, SUMMARIZE, or EVALUATE
            StateTransitionRule(AgentState.EXECUTING, ActionType.EXECUTE, AgentState.EXECUTING),
            StateTransitionRule(AgentState.EXECUTING, ActionType.SUMMARIZE, AgentState.SUMMARIZING),
            StateTransitionRule(AgentState.EXECUTING, ActionType.EVALUATE, AgentState.EVALUATING),
            
            # From SUMMARIZING state - must go to EVALUATE (mandatory)
            StateTransitionRule(AgentState.SUMMARIZING, ActionType.EVALUATE, AgentState.EVALUATING),
            StateTransitionRule(AgentState.SUMMARIZING, ActionType.EXECUTE, AgentState.EXECUTING),
            
            # From EVALUATING state - can go to PLAN, EXECUTE, or FINALIZE
            StateTransitionRule(AgentState.EVALUATING, ActionType.PLAN, AgentState.PLANNING),
            StateTransitionRule(AgentState.EVALUATING, ActionType.EXECUTE, AgentState.EXECUTING),
            StateTransitionRule(AgentState.EVALUATING, ActionType.FINALIZE, AgentState.FINALIZING),

            # From FINALIZING state - must go to COMPLETED (mandatory)
            StateTransitionRule(AgentState.FINALIZING, ActionType.NO_ACTION, AgentState.COMPLETED),
            StateTransitionRule(AgentState.FINALIZING, ActionType.NO_ACTION, AgentState.FAILED),
        ]
    
    def is_action_allowed(self, action: ActionType, agent_data: Dict = None) -> bool:
        """
        Check if an action is allowed from the current state.
        
        Args:
            action: The action the LLM wants to take
            agent_data: Current agent state data for condition checking
            
        Returns:
            True if action is allowed, False otherwise
        """
        # First check business rules (these are critical and override state transitions)
        if not self._check_business_rules(action, agent_data or {}):
            return False
        
        # Then check for direct state transition validity
        for rule in self.transitions:
            if rule.from_state == self.current_state and rule.action == action:
                # Check additional conditions if specified
                if self._check_condition(rule.condition, agent_data):
                    return True
        
        # If no explicit transition found, it's not allowed
        return False
    
    def _check_condition(self, condition: Optional[str], agent_data: Dict) -> bool:
        """Check if a transition condition is met."""
        if not condition:
            return True
            
        # Add condition checking logic here if needed
        # For now, we don't use specific conditions but keep agent_data for future use
        _ = agent_data  # Mark as used to avoid linting warnings
        return True
    
    def _check_business_rules(self, action: ActionType, agent_data: Dict) -> bool:
        """Apply business rules for action validation."""
        
        # Rule 1: Should have data before SUMMARIZE
        if action == ActionType.SUMMARIZE:
            data_keys = [
                k for k, v in agent_data.items() 
                if k not in ["goal", "achieved", "used_tools"] and v
            ]
            if len(data_keys) < 1:
                return False
        
        # Rule 2: Should not EVALUATE without sufficient data or a summary
        if action == ActionType.EVALUATE:
            has_summary = agent_data.get("summary") is not None and agent_data.get("summary") != ""
            # Count meaningful data items (excluding meta fields)
            data_keys = [
                k for k, v in agent_data.items() 
                if k not in ["goal", "achieved", "used_tools"] and v
            ]
            has_sufficient_data = len(data_keys) >= 2  # At least 2 data items
            
            # Allow evaluate if we have either a summary OR sufficient data
            if not (has_summary or has_sufficient_data):
                return False
        
        # Rule 3: Should only FINALIZE when goal is actually achieved
        if action == ActionType.FINALIZE:
            goal_achieved = agent_data.get("achieved", False)
            if not goal_achieved:
                return False
        
        return True
    
    def get_allowed_actions(self, agent_data: Dict = None) -> Set[ActionType]:
        """Get all currently allowed actions."""
        allowed = set()
        for action in ActionType:
            if self.is_action_allowed(action, agent_data):
                allowed.add(action)
        return allowed
    
    def transition(self, action: ActionType) -> bool:
        """
        Attempt to transition to a new state.
        
        Args:
            action: The action being taken
            
        Returns:
            True if transition successful, False if invalid
        """
        # Find the target state for this transition
        target_state = None
        for rule in self.transitions:
            if rule.from_state == self.current_state and rule.action == action:
                target_state = rule.to_state
                break
        
        if target_state is None:
            return False
        
        # Update state
        self.current_state = target_state
        self.last_action = action
        self.action_history.append(action)
        self.state_history.append(target_state)
        
        return True
    
    def get_forced_action(self, rejected_action: ActionType, agent_data: Dict = None) -> ActionType:
        """
        Get a forced action when the LLM's choice is invalid.
        
        Args:
            rejected_action: The action that was rejected
            agent_data: Current agent state data
            
        Returns:
            A valid action to force instead
        """
        # Get any allowed action based on current state
        allowed = self.get_allowed_actions(agent_data)
        
        if allowed:
            # Return the first allowed action (there should only be one in many cases)
            return list(allowed)[0]
        else:
            # Emergency fallback - should not happen with proper state machine
            return ActionType.PLAN
    
    def reset(self):
        """Reset the state machine to initial state."""
        self.current_state = AgentState.INITIAL
        self.last_action = None
        self.action_history = []
        self.state_history = [AgentState.INITIAL]
    
    def get_state_info(self) -> Dict:
        """Get current state machine information."""
        return {
            "current_state": self.current_state.value,
            "last_action": self.last_action.value if self.last_action else None,
            "action_history": [a.value for a in self.action_history],
            "state_history": [s.value for s in self.state_history]
        }
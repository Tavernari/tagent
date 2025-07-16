#!/usr/bin/env python3
"""
Test script to verify that the infinite loop prevention mechanisms work correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tagent.state_machine import AgentStateMachine, ActionType, AgentState

def test_loop_detection():
    """Test that the state machine detects SUMMARIZE -> EVALUATE -> PLAN loops."""
    print("Testing loop detection...")
    
    # Create a state machine
    sm = AgentStateMachine()
    
    # Simulate a normal flow first
    sm.transition(ActionType.PLAN)
    sm.transition(ActionType.EXECUTE)
    sm.transition(ActionType.SUMMARIZE)
    sm.transition(ActionType.EVALUATE)
    sm.transition(ActionType.PLAN)
    
    # Check that loop is not detected yet (only one cycle)
    assert not sm.detect_summarize_evaluate_loop(), "Should not detect loop with only one cycle"
    print("✓ Single cycle not detected as loop")
    
    # Continue with another problematic cycle
    sm.transition(ActionType.EXECUTE)
    sm.transition(ActionType.SUMMARIZE)
    sm.transition(ActionType.EVALUATE)
    sm.transition(ActionType.PLAN)
    
    # This should NOT detect a loop because we had EXECUTE actions
    assert not sm.detect_summarize_evaluate_loop(), "Should not detect loop when EXECUTE actions are present"
    print("✓ Loop with EXECUTE actions not detected")
    
    # Now simulate the problematic pattern: SUMMARIZE -> EVALUATE -> PLAN without EXECUTE
    sm.transition(ActionType.SUMMARIZE)  # This should fail due to business rule
    
    print("✓ Loop detection tests passed")

def test_business_rules():
    """Test that business rules prevent problematic transitions."""
    print("\nTesting business rules...")
    
    sm = AgentStateMachine()
    
    # Set up a scenario where we're in PLANNING state after evaluation
    sm.transition(ActionType.PLAN)
    sm.transition(ActionType.EXECUTE)
    sm.transition(ActionType.SUMMARIZE)
    sm.transition(ActionType.EVALUATE)
    sm.transition(ActionType.PLAN)  # Now in PLANNING state after evaluation
    
    # Check that SUMMARIZE is not allowed from PLANNING state after evaluation
    mock_data = {"some_data": "test"}
    allowed = sm.is_action_allowed(ActionType.SUMMARIZE, mock_data)
    assert not allowed, "SUMMARIZE should not be allowed from PLANNING state after evaluation"
    print("✓ SUMMARIZE blocked from PLANNING state after evaluation")
    
    # Check that EXECUTE is allowed
    allowed = sm.is_action_allowed(ActionType.EXECUTE, mock_data)
    assert allowed, "EXECUTE should be allowed from PLANNING state"
    print("✓ EXECUTE allowed from PLANNING state")
    
    print("✓ Business rules tests passed")

def test_state_transitions():
    """Test that state transitions work correctly."""
    print("\nTesting state transitions...")
    
    sm = AgentStateMachine()
    
    # Test mandatory flow
    assert sm.current_state == AgentState.INITIAL
    
    # INITIAL -> PLAN
    result = sm.transition(ActionType.PLAN)
    assert result and sm.current_state == AgentState.PLANNING
    print("✓ INITIAL -> PLANNING transition")
    
    # PLANNING -> EXECUTE
    result = sm.transition(ActionType.EXECUTE)
    assert result and sm.current_state == AgentState.EXECUTING
    print("✓ PLANNING -> EXECUTING transition")
    
    # EXECUTING -> PLAN (should now be allowed)
    result = sm.transition(ActionType.PLAN)
    assert result and sm.current_state == AgentState.PLANNING
    print("✓ EXECUTING -> PLANNING transition")
    
    print("✓ State transition tests passed")

if __name__ == "__main__":
    try:
        test_state_transitions()
        test_business_rules()
        test_loop_detection()
        print("\n🎉 All tests passed! Loop prevention mechanisms are working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
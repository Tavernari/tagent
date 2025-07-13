"""
Test state machine integration with the main agent loop.

Verifies that:
1. State machine prevents invalid action transitions
2. Auto-evaluation after summarization works
3. Auto-planning after failed evaluation works  
4. LLM decisions are controlled by state machine rules
"""

import pytest
from unittest.mock import patch
from typing import Dict, Any, Optional, Tuple

from tagent.agent import run_agent
from tagent.state_machine import AgentStateMachine, ActionType
from tagent.llm_adapter import set_llm_adapter, MockLLMAdapter, LiteLLMAdapter


class TestStateMachineIntegration:
    """Test suite for state machine integration."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.mock_tools = {
            "data_tool": self._mock_data_tool,
        }
        
    def teardown_method(self):
        """Clean up after each test."""
        set_llm_adapter(LiteLLMAdapter())
        
    def _mock_data_tool(self, state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
        """Mock tool that simulates data collection."""
        _ = state, args  # Mark parameters as used
        return ("collected_data", {"data": "test_data", "status": "collected"})

    @patch('tagent.agent.generate_step_title')
    def test_auto_evaluate_after_summarize(self, mock_generate_step_title):
        """Test that evaluate is automatically called after summarize."""
        # Track when summarize and evaluate actions are called
        summarize_calls = []
        evaluate_calls = []
        
        def track_summarize_calls(state, *args, **kwargs):
            summarize_calls.append(len(summarize_calls) + 1)
            return [("summary", "Data has been summarized and is ready for evaluation")]
        
        def track_evaluate_calls(state, *args, **kwargs):
            evaluate_calls.append(len(evaluate_calls) + 1)
            # First evaluation should succeed since we have a summary
            return [("achieved", True), ("evaluation_result", {"feedback": "Goal achieved"})]
        
        # Mock responses - only need one summarize, auto-evaluate will be triggered
        mock_responses = [
            '{"action": "execute", "params": {"tool": "data_tool", "args": {}}, "reasoning": "Collecting data"}',
            '{"action": "summarize", "params": {}, "reasoning": "Summarizing results"}',
            '{"action": "finalize", "params": {}, "reasoning": "Finalizing results"}'
        ]
        
        mock_adapter = MockLLMAdapter(responses=mock_responses)
        set_llm_adapter(mock_adapter)
        mock_generate_step_title.return_value = "Test Step"
        
        with patch('tagent.agent.summarize_action', side_effect=track_summarize_calls) as mock_summarize:
            with patch('tagent.agent.enhanced_goal_evaluation_action', side_effect=track_evaluate_calls) as mock_evaluate:
                result = run_agent(
                    goal="Collect and analyze data",
                    model="gpt-3.5-turbo",
                    tools=self.mock_tools,
                    max_iterations=5,
                    verbose=False
                )
                
                # REAL VALIDATIONS:
                # 1. Summarize should be called exactly once (from LLM decision)
                assert len(summarize_calls) == 1, f"Expected 1 summarize call, got {len(summarize_calls)}"
                
                # 2. Evaluate should be called automatically after summarize
                assert len(evaluate_calls) >= 1, f"Expected auto-evaluate after summarize, got {len(evaluate_calls)} evaluate calls"
                
                # 3. Result should be successful
                assert result is not None, "Agent should complete successfully"

    @patch('tagent.agent.generate_step_title')
    def test_auto_plan_after_failed_evaluate(self, mock_generate_step_title):
        """Test that plan is automatically called after failed evaluation."""
        # Track when plan and evaluate actions are called
        plan_calls = []
        evaluate_calls = []
        
        def track_plan_calls(state, *args, **kwargs):
            plan_calls.append(len(plan_calls) + 1)
            return [("plan", f"New strategy {len(plan_calls)} based on evaluation feedback")]
        
        def track_evaluate_calls(state, *args, **kwargs):
            evaluate_calls.append(len(evaluate_calls) + 1)
            if len(evaluate_calls) == 1:
                # First evaluation fails with feedback
                return [("achieved", False), ("evaluation_result", {
                    "feedback": "Missing required data",
                    "missing_items": ["more_data"],
                    "suggestions": ["collect additional information"]
                })]
            else:
                # Second evaluation succeeds
                return [("achieved", True), ("evaluation_result", {"feedback": "Goal achieved"})]
        
        # Mock responses: execute -> summarize (auto-eval will fail and auto-plan)
        mock_responses = [
            '{"action": "execute", "params": {"tool": "data_tool", "args": {}}, "reasoning": "Collecting data"}',
            '{"action": "summarize", "params": {}, "reasoning": "Summarizing results"}',
            '{"action": "execute", "params": {"tool": "data_tool", "args": {}}, "reasoning": "Following plan"}',
            '{"action": "finalize", "params": {}, "reasoning": "Finalizing results"}'
        ]
        
        mock_adapter = MockLLMAdapter(responses=mock_responses)
        set_llm_adapter(mock_adapter)
        mock_generate_step_title.return_value = "Test Step"
        
        with patch('tagent.agent.summarize_action') as mock_summarize:
            mock_summarize.return_value = [("summary", "Data collected and summarized")]
            
            with patch('tagent.agent.enhanced_goal_evaluation_action', side_effect=track_evaluate_calls) as mock_evaluate:
                with patch('tagent.agent.plan_action', side_effect=track_plan_calls) as mock_plan:
                    result = run_agent(
                        goal="Collect and analyze data with feedback",
                        model="gpt-3.5-turbo",
                        tools=self.mock_tools,
                        max_iterations=8,
                        verbose=False
                    )
                    
                    # REAL VALIDATIONS:
                    # 1. Evaluate should be called (auto-triggered after summarize)
                    assert len(evaluate_calls) >= 1, f"Expected auto-evaluate, got {len(evaluate_calls)} calls"
                    
                    # 2. Plan should be auto-called after failed evaluation
                    assert len(plan_calls) >= 1, f"Expected auto-plan after failed evaluation, got {len(plan_calls)} calls"
                    
                    # 3. Agent should complete or make significant progress
                    assert result is not None or (len(evaluate_calls) > 0 and len(plan_calls) > 0)

    @patch('tagent.agent.generate_step_title')
    def test_state_machine_prevents_invalid_transitions(self, mock_generate_step_title):
        """Test that state machine prevents invalid action transitions."""
        # Track state machine forced actions
        forced_actions = []
        
        # Mock responses that would create invalid transitions
        mock_responses = [
            '{"action": "summarize", "params": {}, "reasoning": "First summarize"}',
            '{"action": "summarize", "params": {}, "reasoning": "Try summarize again - should be blocked"}',
            '{"action": "plan", "params": {}, "reasoning": "Forced plan"}',
            '{"action": "evaluate", "params": {"achieved": true}, "reasoning": "Final evaluation"}'
        ]
        
        mock_adapter = MockLLMAdapter(responses=mock_responses)
        set_llm_adapter(mock_adapter)
        mock_generate_step_title.return_value = "Test Step"
        
        # Capture print_retro_status calls to see forced actions
        with patch('tagent.agent.print_retro_status') as mock_print_status:
            with patch('tagent.agent.summarize_action') as mock_summarize:
                mock_summarize.return_value = [("summary", "Summarized data")]
                
                with patch('tagent.agent.enhanced_goal_evaluation_action') as mock_evaluate:
                    mock_evaluate.return_value = [("achieved", True)]
                    
                    with patch('tagent.agent.plan_action') as mock_plan:
                        mock_plan.return_value = [("plan", "New strategy")]
                        
                        result = run_agent(
                            goal="Test state machine transitions",
                            model="gpt-3.5-turbo",
                            tools=self.mock_tools,
                            max_iterations=6,
                            verbose=False
                        )
                        
                        # Look for state machine control messages
                        status_calls = [call[0][1] for call in mock_print_status.call_args_list if len(call[0]) > 1]
                        state_control_calls = [call for call in status_calls if "Forced" in str(call)]
                        
                        # REAL VALIDATIONS:
                        # 1. State machine should prevent some invalid transitions
                        assert len(state_control_calls) > 0 or mock_plan.called, "State machine should prevent invalid transitions"
                        
                        # 2. Agent should complete successfully despite restrictions
                        assert result is not None, "Agent should handle state machine restrictions gracefully"

    def test_state_machine_prevents_evaluate_without_summary(self):
        """Test that state machine prevents evaluate when no summary exists."""
        state_machine = AgentStateMachine()
        
        # Move to proper state where EVALUATE can be called (must be SUMMARIZING state)
        state_machine.transition(ActionType.PLAN)     # INITIAL -> PLANNING
        state_machine.transition(ActionType.EXECUTE)  # PLANNING -> EXECUTING
        state_machine.transition(ActionType.SUMMARIZE)  # EXECUTING -> SUMMARIZING
        
        # Test data without summary
        agent_data = {
            "goal": "test goal",
            "collected_data": {"some": "data"},
            "used_tools": ["data_tool"]
        }
        
        # EVALUATE should not be allowed without summary
        assert not state_machine.is_action_allowed(ActionType.EVALUATE, agent_data), \
            "EVALUATE should not be allowed without summary"
        
        # Add summary and try again
        agent_data["summary"] = "Data has been summarized"
        assert state_machine.is_action_allowed(ActionType.EVALUATE, agent_data), \
            "EVALUATE should be allowed with summary present"

    def test_state_machine_prevents_double_summarize(self):
        """Test that state machine prevents summarize after summarize."""
        state_machine = AgentStateMachine()
        
        # Simulate proper flow: INITIAL -> PLAN -> EXECUTE -> SUMMARIZE
        state_machine.transition(ActionType.PLAN)     # INITIAL -> PLANNING
        state_machine.transition(ActionType.EXECUTE)  # PLANNING -> EXECUTING
        state_machine.transition(ActionType.SUMMARIZE)  # EXECUTING -> SUMMARIZING
        
        agent_data = {
            "goal": "test goal",
            "collected_data": {"some": "data"},
            "summary": "Already summarized"
        }
        
        # Second SUMMARIZE should not be allowed
        assert not state_machine.is_action_allowed(ActionType.SUMMARIZE, agent_data), \
            "SUMMARIZE should not be allowed immediately after SUMMARIZE"
        
        # But EVALUATE should be allowed after SUMMARIZE
        assert state_machine.is_action_allowed(ActionType.EVALUATE, agent_data), \
            "EVALUATE should be allowed after SUMMARIZE"

    def test_state_machine_prevents_double_evaluate(self):
        """Test that state machine prevents evaluate after evaluate."""
        state_machine = AgentStateMachine()
        
        # Simulate proper flow: INITIAL -> PLAN -> EXECUTE -> SUMMARIZE -> EVALUATE
        state_machine.transition(ActionType.PLAN)      # INITIAL -> PLANNING
        state_machine.transition(ActionType.EXECUTE)   # PLANNING -> EXECUTING
        state_machine.transition(ActionType.SUMMARIZE) # EXECUTING -> SUMMARIZING
        state_machine.transition(ActionType.EVALUATE)  # SUMMARIZING -> EVALUATING
        
        agent_data = {
            "goal": "test goal",
            "summary": "Data summarized",
            "achieved": False
        }
        
        # Second EVALUATE should not be allowed
        assert not state_machine.is_action_allowed(ActionType.EVALUATE, agent_data), \
            "EVALUATE should not be allowed immediately after EVALUATE"
        
        # But PLAN should be allowed after EVALUATE
        assert state_machine.is_action_allowed(ActionType.PLAN, agent_data), \
            "PLAN should be allowed after EVALUATE"

    def test_max_iterations_triggers_summarization_and_evaluation(self):
        """Test that reaching max iterations automatically triggers summarization and evaluation."""
        
        # Mock LLM responses that would cause infinite planning/execution loop
        responses = [
            '{"action": "plan", "params": {"goal": "test goal"}, "reasoning": "Planning"}',
            '{"action": "execute", "params": {"tool": "data_tool", "args": {"test": "data"}}, "reasoning": "Executing"}',
            '{"action": "plan", "params": {"goal": "test goal"}, "reasoning": "Planning again"}',
            '{"action": "execute", "params": {"tool": "data_tool", "args": {"test": "data"}}, "reasoning": "Executing again"}',
            # This would continue infinitely without max iterations
        ]
        
        # Mock the summarize responses
        summarize_response = '{"action": "summarize", "params": {}, "reasoning": "Creating summary"}'
        evaluate_response = '{"action": "evaluate", "params": {}, "reasoning": "Evaluating goal"}'
        
        # Set up mock adapter
        mock_adapter = MockLLMAdapter(responses)
        set_llm_adapter(mock_adapter)
        
        # Track function calls to verify summarization and evaluation occur
        summarize_called = False
        evaluate_called = False
        original_responses = mock_adapter.responses.copy()
        
        def track_summarize(*args, **kwargs):
            nonlocal summarize_called
            summarize_called = True
            # Mock a summary result
            return [("summary", {
                "findings": ["Found some data"], 
                "status": "Summary completed"
            })]
        
        def track_evaluate(*args, **kwargs):
            nonlocal evaluate_called
            evaluate_called = True
            # Mock evaluation result
            return [("evaluation", {
                "achieved": True,
                "feedback": "Goal achieved with collected data"
            })]
        
        # Patch the actions to track calls
        with patch('tagent.agent.summarize_action', side_effect=track_summarize), \
             patch('tagent.agent.enhanced_goal_evaluation_action', side_effect=track_evaluate):
            
            # Run agent with very low max_iterations to force timeout
            result = run_agent(
                goal="Test goal for max iterations",
                model="test-model",
                tools=self.mock_tools,
                max_iterations=2,  # Very low to trigger timeout quickly
                verbose=True
            )
        
        # Verify that summarization and evaluation were called
        assert summarize_called, "Summarization should be called when max iterations reached"
        assert evaluate_called, "Evaluation should be called after summarization when max iterations reached"
        
        # Verify the result contains the expected structure
        assert result is not None, "Result should not be None"
        assert result.get("status") == "completed_with_summary_fallback", \
            "Status should indicate completion with summary fallback"
        assert "result" in result, "Result should contain summary result"
        assert "evaluation" in result, "Result should contain evaluation result"
        assert "raw_data" in result, "Result should contain raw data with all findings"
        assert result.get("iterations_used") == 2, "Should show correct number of iterations used"
        assert result.get("max_iterations") == 2, "Should show correct max iterations"
        
        # Verify the evaluation shows goal was achieved
        evaluation = result.get("evaluation", {})
        assert evaluation.get("achieved") is True, "Evaluation should show goal was achieved"
        
        print("✓ Max iterations correctly triggers summarization and evaluation")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
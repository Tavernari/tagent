"""
Task-based TAgent - Agent that executes tasks in loops with retry logic.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Type, Callable, TypeVar

try:
    from typing import Generic
except ImportError:
    from typing_extensions import Generic

from pydantic import BaseModel
import litellm

from .state_machine import TaskBasedStateMachine, AgentPhase
from .memory_manager import EnhancedContextManager
from .task_actions import (
    task_based_plan_action,
    task_based_execute_action,
    task_based_evaluate_action,
    task_based_finalize_action
)
from .ui import (
    print_retro_banner,
    print_retro_status,
    print_retro_step,
    start_thinking,
    stop_thinking,
    MessageType
)
from .version import __version__
from .instructions_rag import InstructionsRAG
from .tool_rag import ToolRAG
from .prompt_builder import PromptBuilder
from .models import TokenStats

# Define a TypeVar for the output model
OutputType = TypeVar("OutputType", bound=Optional[BaseModel])


class TaskBasedAgentResult(Generic[OutputType], BaseModel):
    """Result from task-based agent execution."""
    
    output: OutputType = None
    goal_achieved: bool = False
    iterations_used: int = 0
    planning_cycles: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    state_summary: Dict[str, Any] = {}
    memory_summary: Dict[str, Any] = {}
    failure_reason: Optional[str] = None
    stats: Optional[TokenStats] = None

    class Config:
        arbitrary_types_allowed = True


def _detect_language(text: str, api_key: Optional[str] = None) -> str:
    """Detect the language of a given text."""
    try:
        prompt = f"What language is the following text written in? Respond with the language name only (e.g., 'English', 'Portuguese').\n\nText: '{text}'"
        messages = [{"role": "user", "content": prompt}]
        response = litellm.completion(
            model="gemini/gemini-pro",
            messages=messages,
            temperature=0.0,
            api_key=api_key
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "English"  # Default to English on failure


def _translate_text(text: str, target_language: str, api_key: Optional[str] = None) -> str:
    """Translate text to a target language."""
    if not text:
        return ""
    try:
        prompt = f"Translate the following text to {target_language}. Only return the translated text, with no additional comments or explanations:\n\n{text}"
        messages = [{"role": "user", "content": prompt}]
        response = litellm.completion(
            model="gemini/gemini-pro",
            messages=messages,
            temperature=0.1,
            api_key=api_key
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: Translation failed. Returning original text. Error: {e}")
        return text



def run_task_based_agent(
    goal: str,
    tools: Dict[str, Callable],
    output_format: Optional[Type[OutputType]] = None,
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    max_iterations: int = 20,
    max_planning_cycles: int = 5,
    verbose: bool = False
) -> TaskBasedAgentResult[OutputType]:
    """
    Run TAgent with task-based execution and retry logic.
    
    Args:
        goal: The objective to achieve
        tools: Dictionary of available tools
        output_format: Desired output format (Pydantic model)
        model: LLM model to use
        api_key: API key for LLM
        max_iterations: Maximum iterations allowed
        max_planning_cycles: Maximum planning cycles
        verbose: Whether to show debug output
        
    Returns:
        TaskBasedAgentResult with comprehensive execution data
    """
    # --- Language Standardization ---
    original_language = _detect_language(goal, api_key)
    if original_language.lower() != "english":
        english_goal = _translate_text(goal, "English", api_key)
        if verbose:
            print_retro_status("TRANSLATE", f"Goal translated from {original_language} to English.")
    else:
        english_goal = goal
    
    # Initialize components
    tool_names = list(tools.keys())
    state_machine = TaskBasedStateMachine(english_goal, tool_names)
    state_machine.max_iterations = max_iterations
    state_machine.max_planning_cycles = max_planning_cycles
    
    context_manager = EnhancedContextManager(english_goal)
    tool_rag = ToolRAG(tools)
    instructions_rag = InstructionsRAG(tool_rag=tool_rag)  # Inject ToolRAG
    prompt_builder = PromptBuilder(instructions_rag)
    token_stats = TokenStats()  # Initialize token usage tracking
    
    # Initialize UI
    print_retro_banner(f"TASK-BASED T-AGENT v{__version__} STARTING", "─", 60, MessageType.PRIMARY)
    print_retro_status("INIT", f"Goal: {goal[:50]}...")
    print_retro_status("CONFIG", f"Model: {model} | Max Iterations: {max_iterations}")
    print_retro_status("CONFIG", f"Max Planning Cycles: {max_planning_cycles}")
    print_retro_status("TOOLS", f"Available: {', '.join(tool_names)}")
    
    # Main execution loop
    step_count = 0
    
    while state_machine.should_continue():
        current_phase = state_machine.get_current_phase()
        step_count += 1
        
        print_retro_step(
            step_count,
            current_phase.value,
            f"Phase {current_phase.value} (cycle {state_machine.planning_cycles})"
        )
        
        if verbose:
            context = state_machine.get_execution_context()
            print(f"[DEBUG] Context: {context}")
        
        # Execute phase-specific actions
        if current_phase == AgentPhase.INIT:
            # Initialize and transition to planning
            print_retro_status("INIT", "Initializing task-based agent...")
            state_machine.set_phase(AgentPhase.PLANNING)
            
        elif current_phase == AgentPhase.PLANNING:
            _execute_planning_phase(state_machine, context_manager, prompt_builder, model, api_key, verbose, token_stats)
            
        elif current_phase == AgentPhase.EXECUTING:
            _execute_task_loop(state_machine, context_manager, prompt_builder, tool_rag, tools, model, api_key, verbose, token_stats)
            
        elif current_phase == AgentPhase.EVALUATING:
            _execute_evaluation_phase(state_machine, context_manager, prompt_builder, model, api_key, verbose, token_stats)
            
        elif current_phase in [AgentPhase.COMPLETED, AgentPhase.FAILED]:
            break
            
        else:
            print_retro_status("ERROR", f"Unknown phase: {current_phase}")
            break
    
    # Final results
    final_phase = state_machine.get_current_phase()
    goal_achieved = final_phase == AgentPhase.COMPLETED
    
    if goal_achieved:
        print_retro_banner("MISSION COMPLETE", "─", 60, MessageType.SUCCESS)
        print_retro_status("SUCCESS", "Goal achieved successfully!")
    else:
        print_retro_banner("MISSION ENDED", "─", 60, MessageType.WARNING)
        print_retro_status("INFO", f"Stopped at phase: {final_phase.value}")
    
    # Get final output
    final_output = None
    if goal_achieved:
        final_output = _execute_finalize_phase(state_machine, context_manager, prompt_builder, output_format, model, api_key, verbose, token_stats)

        # --- Translate final output back to original language ---
        if original_language.lower() != "english" and final_output:
            if isinstance(final_output, BaseModel):
                # If the output is a Pydantic model, translate its string fields
                translated_data = {}
                for field_name, field_value in final_output:
                    if isinstance(field_value, str):
                        translated_data[field_name] = _translate_text(field_value, original_language, api_key)
                    else:
                        translated_data[field_name] = field_value
                final_output = type(final_output)(**translated_data)
                if verbose:
                    print_retro_status("TRANSLATE", f"Final output translated back to {original_language}.")
            elif isinstance(final_output, str):
                final_output = _translate_text(final_output, original_language, api_key)

    # Compile results
    task_summary = state_machine.get_task_summary()
    
    result = TaskBasedAgentResult(
        output=final_output,
        goal_achieved=goal_achieved,
        iterations_used=state_machine.current_iteration,
        planning_cycles=state_machine.planning_cycles,
        total_tasks=task_summary['total_tasks'],
        completed_tasks=task_summary['completed'],
        failed_tasks=task_summary['failed'],
        state_summary=state_machine.get_state_summary(),
        memory_summary=context_manager.get_memory_summary_for_finalize(),
        failure_reason=state_machine.state.failure_reason,
        stats=token_stats
    )
    
    # Display final token statistics
    if token_stats.total.total_tokens > 0:
        from .ui import print_feedback_dimmed
        
        print_feedback_dimmed("TOKEN_USAGE", "Token Usage Breakdown:")
        
        # Sort models by name for consistent output
        for model_name, usage in sorted(token_stats.by_model.items()):
            model_cost_str = f"${usage.cost:.6f}" if usage.cost > 0 else "$0.000000"
            model_summary = (
                f"  - {model_name}: "
                f"Input: {usage.input_tokens}, Output: {usage.output_tokens}, "
                f"Total: {usage.total_tokens} tokens ({model_cost_str})"
            )
            print_feedback_dimmed("MODEL_STATS", model_summary)
            
        cost_str = f"${token_stats.total.cost:.6f}" if token_stats.total.cost > 0 else "$0.000000"
        total_summary = (
            f"Grand Total: "
            f"Input: {token_stats.total.input_tokens}, Output: {token_stats.total.output_tokens}, "
            f"Total: {token_stats.total.total_tokens} tokens ({cost_str})"
        )
        print_feedback_dimmed("TOTAL_STATS", total_summary)
    
    if verbose:
        print(f"[FINAL] Result: {result}")
    
    return result


def _execute_planning_phase(
    state_machine: TaskBasedStateMachine,
    context_manager: EnhancedContextManager,
    prompt_builder: PromptBuilder,
    model: str,
    api_key: Optional[str],
    verbose: bool,
    token_stats: TokenStats
) -> bool:
    """Execute the planning phase."""
    planning_cycle = state_machine.planning_cycles
    
    if planning_cycle > 0:
        print_retro_status("REPLAN", f"Re-planning cycle {planning_cycle} with failure context...")
    else:
        print_retro_status("PLAN", "Initial planning with RAG context...")
    
    response = task_based_plan_action(
        state_machine=state_machine,
        context_manager=context_manager,
        prompt_builder=prompt_builder,
        model=model,
        api_key=api_key,
        verbose=verbose,
        token_stats=token_stats
    )
    
    if response:
        task_count = len(response.tasks)
        print_retro_status("SUCCESS", f"Plan created with {task_count} tasks")
        
        if verbose:
            print(f"[PLAN] Strategy: {response.strategy}")
            for i, task in enumerate(response.tasks, 1):
                print(f"[PLAN] Task {i}: {task.description} → {task.tool_name}")
        
        # Move to execution phase
        state_machine.set_phase(AgentPhase.EXECUTING)
        return True
    else:
        print_retro_status("ERROR", "Planning failed")
        state_machine.mark_completed(False)
        return False


def _execute_task_loop(
    state_machine: TaskBasedStateMachine,
    context_manager: EnhancedContextManager,
    prompt_builder: PromptBuilder,
    tool_rag: ToolRAG,
    tools: Dict[str, Callable],
    model: str,
    api_key: Optional[str],
    verbose: bool,
    token_stats: TokenStats
) -> bool:
    """Execute the task loop with retry logic."""
    print_retro_status("EXECUTE", "Starting task execution loop...")
    
    task_count = 0
    
    # Execute tasks until all are processed
    while state_machine.has_pending_tasks():
        task_count += 1
        current_task = state_machine.get_current_task()
        
        if not current_task:
            break
        
        # Show task execution
        retry_info = f" (retry {current_task.retry_count})" if current_task.retry_count > 0 else ""
        print_retro_status("TASK", f"[{task_count}] {current_task.description}{retry_info}")
        
        # Execute the task
        response = task_based_execute_action(
            state_machine=state_machine,
            context_manager=context_manager,
            prompt_builder=prompt_builder,
            tool_rag=tool_rag,
            tools=tools,
            model=model,
            api_key=api_key,
            verbose=verbose,
            token_stats=token_stats
        )
        
        if response and response.token_usage and token_stats:
            from .ui import print_feedback_dimmed
            token_stats.add_usage(response.token_usage)
            print_feedback_dimmed("TOKENS", token_stats.format_dimmed_summary(model))
        
        if verbose:
            print(f"[DEBUG] Response from execute action: {response}")
        
        if response:
            if response.success:
                print_retro_status("SUCCESS", f"Task completed: {current_task.description}")
            else:
                print_retro_status("WARNING", f"Task failed: {response.failure_reason}")
        else:
            print_retro_status("ERROR", f"Task execution error: {current_task.description}")
    
    # Show task summary
    task_summary = state_machine.get_task_summary()
    print_retro_status("SUMMARY", f"Tasks: {task_summary['completed']} completed, {task_summary['failed']} failed")
    
    # Move to evaluation phase
    state_machine.set_phase(AgentPhase.EVALUATING)
    return True


def _execute_evaluation_phase(
    state_machine: TaskBasedStateMachine,
    context_manager: EnhancedContextManager,
    prompt_builder: PromptBuilder,
    model: str,
    api_key: Optional[str],
    verbose: bool,
    token_stats: TokenStats
) -> bool:
    """Execute the evaluation phase."""
    print_retro_status("EVALUATE", "Evaluating goal achievement...")
    
    response = task_based_evaluate_action(
        state_machine=state_machine,
        context_manager=context_manager,
        prompt_builder=prompt_builder,
        model=model,
        api_key=api_key,
        verbose=verbose,
        token_stats=token_stats
    )
    
    if response:
        if response.goal_achieved:
            print_retro_status("SUCCESS", f"Goal achieved! (confidence: {response.confidence:.1%})")
            # State machine already marked as completed
            return True
        else:
            print_retro_status("INFO", f"Goal not achieved: {response.failure_reason}")
            
            # Check if we should return to planning
            if state_machine.get_current_phase() == AgentPhase.PLANNING:
                print_retro_status("RETRY", "Returning to planning with failure context...")
                return True
            else:
                print_retro_status("FINAL", "Max planning cycles reached")
                return False
    else:
        print_retro_status("ERROR", "Evaluation failed")
        return False


def _execute_finalize_phase(
    state_machine: TaskBasedStateMachine,
    context_manager: EnhancedContextManager,
    prompt_builder: PromptBuilder,
    output_format: Optional[Type[BaseModel]],
    model: str,
    api_key: Optional[str],
    verbose: bool,
    token_stats: TokenStats
) -> Any:
    """Execute the finalization phase."""
    print_retro_status("FINALIZE", "Creating final output from all tasks and memories...")
    
    final_output = task_based_finalize_action(
        state_machine=state_machine,
        context_manager=context_manager,
        prompt_builder=prompt_builder,
        output_format=output_format,
        model=model,
        api_key=api_key,
        verbose=verbose,
        token_stats=token_stats
    )
    
    if final_output:
        print_retro_status("SUCCESS", "Final output created successfully")
        if verbose:
            print(f"[FINALIZE] Output: {final_output}")
    else:
        print_retro_status("ERROR", "Finalization failed")
    
    return final_output


# Example usage
if __name__ == "__main__":
    # Example tools
    def example_tool(state: Dict[str, Any], args: Dict[str, Any]):
        # Note: state and args parameters are available for tool logic
        return ("example_result", {"data": "test"})
    
    tools = {"example_tool": example_tool}
    
    # Example output format
    class ExampleOutput(BaseModel):
        summary: str
        tasks_completed: int
        data: Dict[str, Any]
    
    # Run task-based agent
    result = run_task_based_agent(
        goal="Test the task-based agent system",
        tools=tools,
        output_format=ExampleOutput,
        verbose=True
    )
    
    print(f"Result: {result}")

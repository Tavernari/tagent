"""
Task-based actions for TAgent - Actions that work with task-based state machine.
"""

import re
from typing import Dict, Any, List, Optional, Type
from .state_machine import TaskBasedStateMachine


# Helper to resolve nested data
def _get_value_from_path(data: Any, path: str) -> Any:
    """
    Retrieves a value from a nested structure using a dot-separated path.
    Handles dictionary keys, object attributes, and list indices.
    Example path: "articles[0].url"
    """
    # This regex splits the path by dots or brackets, capturing the indices
    keys = re.split(r'\.|\[(\d+)\]', path)
    # Filter out empty strings that result from the split
    keys = [k for k in keys if k]

    current_data = data
    for key in keys:
        if current_data is None:
            return None
            
        if key.isdigit():  # It's an index for a list
            try:
                current_data = current_data[int(key)]
            except (IndexError, TypeError):
                return None  # Return None if index is out of bounds or not a list
        else:  # It's a key for a dict or an attribute for an object
            try:
                if isinstance(current_data, dict):
                    current_data = current_data.get(key)
                else:
                    current_data = getattr(current_data, key, None)
            except (AttributeError, TypeError):
                return None # Return None if attribute/key does not exist
            
    return current_data

def _resolve_dynamic_args(args: Dict[str, Any], state_machine: TaskBasedStateMachine) -> Dict[str, Any]:
    """
    Resolves dynamic arguments in the format `{{tasks.ID.output...}}`
    using regex substitution and path evaluation.
    """
    resolved_args = {}

    def replacer(match: re.Match) -> str:
        """
        This function is called for each placeholder found by re.sub.
        It looks up the task result and returns the appropriate value.
        """
        placeholder = match.group(0)
        expression = match.group(1).strip()  # e.g., "tasks.0.output.articles[0].url"

        # Expression should be like "tasks.INDEX.output.REST_OF_PATH"
        parts = expression.split('.')
        if len(parts) < 3 or parts[0] != 'tasks' or parts[2] != 'output':
            return placeholder

        try:
            task_index = int(parts[1])
            # The state machine uses 1-based IDs like "task_1"
            task_id = f"task_{task_index + 1}"
            
            source_task = next((t for t in state_machine.tasks if t.id == task_id), None)

            if not source_task or source_task.result is None:
                return placeholder

            # result is a tuple like ('key', data)
            result_data = source_task.result[1]
            
            # The rest of the path to resolve
            value_path = '.'.join(parts[3:])
            if not value_path:  # Direct reference like {{tasks.1.output}}
                if isinstance(result_data, dict) and 'response' in result_data:
                    return str(result_data['response'])
                return str(result_data)

            resolved_value = _get_value_from_path(result_data, value_path)

            return str(resolved_value) if resolved_value is not None else placeholder

        except (ValueError, TypeError, AttributeError, IndexError):
            return placeholder

    for key, value in args.items():
        if isinstance(value, str):
            # Regex to find {{...}} placeholders
            resolved_value = re.sub(r"\{\{(.*?)\}\}", replacer, value)
            resolved_args[key] = resolved_value
        else:
            resolved_args[key] = value
            
    return resolved_args


from pydantic import BaseModel, Field
from .models import MemoryItem, TokenStats, TokenUsage
from .memory_manager import EnhancedContextManager
from .state_machine import TaskBasedStateMachine, AgentPhase
from .llm_interface import query_llm_for_model
from .prompt_builder import PromptBuilder
from .tool_rag import ToolRAG
from .tool_executor import ToolExecutor
from .ui import print_feedback_dimmed, start_thinking, stop_thinking


class TaskDefinition(BaseModel):
    """Definition of a task to be executed."""
    description: str = Field(description="Clear description of what the task does")
    tool_name: Optional[str] = Field(default=None, description="Name of the tool to use")
    tool_args: Dict[str, Any] = Field(description="Arguments for the tool")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class TaskBasedPlanResponse(BaseModel):
    """Response for task-based planning."""
    tasks: List[TaskDefinition] = Field(description="List of tasks to execute")
    strategy: str = Field(description="Overall strategy for achieving the goal")
    reasoning: str = Field(description="Reasoning behind the plan")
    memories: List[MemoryItem] = Field(default_factory=list, description="Memories from planning")
    focus_area: Optional[str] = Field(default=None, description="Main focus area")


class TaskBasedExecuteResponse(BaseModel):
    """Response for task-based execution."""
    success: bool = Field(description="Whether execution was successful")
    result: Any = Field(description="Result of the task execution")
    reasoning: str = Field(description="Reasoning behind the execution")
    memories: List[MemoryItem] = Field(default_factory=list, description="Memories from execution")
    failure_reason: Optional[str] = Field(default=None, description="Reason for failure if unsuccessful")
    token_usage: Optional[TokenUsage] = Field(default=None, description="Token usage for this execution")


class TaskBasedEvaluateResponse(BaseModel):
    """Response for task-based evaluation."""
    goal_achieved: bool = Field(description="Whether the goal has been achieved")
    reasoning: str = Field(description="Reasoning behind the evaluation")
    completed_tasks_analysis: str = Field(description="Analysis of completed tasks")
    failed_tasks_analysis: str = Field(description="Analysis of failed tasks")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    confidence: float = Field(default=0.0, description="Confidence in the evaluation")
    memories: List[MemoryItem] = Field(default_factory=list, description="Memories from evaluation")
    failure_reason: Optional[str] = Field(default=None, description="Reason if goal not achieved")


def task_based_plan_action(
    state_machine: TaskBasedStateMachine,
    context_manager: EnhancedContextManager,
    prompt_builder: PromptBuilder,
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    verbose: bool = False,
    token_stats: Optional[TokenStats] = None
) -> Optional[TaskBasedPlanResponse]:
    """
    Planning action that generates a list of tasks to execute.
    
    Args:
        state_machine: Task-based state machine
        context_manager: RAG context manager
        prompt_builder: PromptBuilder instance
        model: LLM model to use
        api_key: API key for LLM
        verbose: Whether to show debug output
        
    Returns:
        Planning response with list of tasks
    """
    state_machine.set_phase(AgentPhase.PLANNING)
    
    # Build prompt using PromptBuilder
    prompt = prompt_builder.build_planning_prompt(state_machine, context_manager)
    
    if verbose:
        print(f"[TASK_PLAN] Generated prompt: {prompt}")
    
    try:
        # Query LLM
        start_thinking("Planning strategy...")
        response = query_llm_for_model(
            prompt=prompt,
            model=model,
            output_model=TaskBasedPlanResponse,
            api_key=api_key,
            verbose=verbose
        )
        stop_thinking()
        
        # Capture token usage and log it dimmed
        if hasattr(query_llm_for_model, '_last_token_usage') and query_llm_for_model._last_token_usage:
            if token_stats:
                token_stats.add_usage(query_llm_for_model._last_token_usage)
                print_feedback_dimmed("TOKENS", token_stats.format_dimmed_summary(model))
        
        # Add tasks to state machine
        tasks_data = []
        for task_def in response.tasks:
            tasks_data.append({
                "description": task_def.description,
                "tool_name": task_def.tool_name,
                "tool_args": task_def.tool_args,
                "max_retries": task_def.max_retries
            })
        
        state_machine.add_tasks(tasks_data)
        
        # Store memories
        if response.memories:
            context_manager.store_memories(response.memories, "task_planning")
        
        # Store execution result
        context_manager.store_execution_result(state_machine.state, "plan", response, True)
        
        return response
        
    except Exception as e:
        if verbose:
            print(f"[TASK_PLAN] Error: {e}")
        
        # Store failure
        context_manager.store_execution_result(state_machine.state, "plan", str(e), False)
        return None


def task_based_execute_action(
    state_machine: TaskBasedStateMachine,
    context_manager: EnhancedContextManager,
    prompt_builder: PromptBuilder,
    tool_rag: ToolRAG,
    tools: Dict[str, Any],
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    verbose: bool = False,
    token_stats: Optional[TokenStats] = None
) -> Optional[TaskBasedExecuteResponse]:
    """
    Execute a single task with retry logic.
    
    Args:
        state_machine: Task-based state machine
        context_manager: RAG context manager
        prompt_builder: PromptBuilder instance
        tool_rag: ToolRAG instance for dynamic tool finding
        tools: Available tools
        model: LLM model to use
        api_key: API key for LLM
        verbose: Whether to show debug output
        
    Returns:
        Execution response with result or failure reason
    """
    state_machine.set_phase(AgentPhase.EXECUTING)
    
    # Get current task
    current_task = state_machine.get_current_task()
    if not current_task:
        return None

    tool_name = current_task.tool_name
    # If no tool is assigned, fallback to llm_task for general reasoning.
    if not tool_name or tool_name == "None":
        tool_name = "llm_task"
        if verbose:
            print(f"[TASK_EXECUTE] No tool assigned for task: '{current_task.description}'. Falling back to 'llm_task'.")

    if verbose:
        print(f"[TASK_EXECUTE] Executing task: {current_task.description}")
        print(f"[TASK_EXECUTE] Tool: {tool_name}")
        print(f"[TASK_EXECUTE] Args: {current_task.tool_args}")

    # Resolve dynamic arguments before execution
    try:
        resolved_args = _resolve_dynamic_args(current_task.tool_args, state_machine)
        if verbose and resolved_args != current_task.tool_args:
            print(f"[TASK_EXECUTE] Resolved dynamic arguments: {resolved_args}")
    except Exception as e:
        failure_reason = f"Failed to resolve dynamic arguments for task '{current_task.description}': {e}"
        state_machine.fail_current_task(failure_reason)
        return TaskBasedExecuteResponse(success=False, result=None, reasoning=failure_reason, failure_reason=failure_reason)

    # Execute the specified tool or the llm_task
    try:
        result = None
        token_usage = None
        if tool_name == "llm_task":
            # Execute the task using the LLM directly
            prompt = resolved_args.get("prompt", current_task.description)
            import litellm
            from .models import TokenUsage
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Complete the user's request directly and concisely."},
                {"role": "user", "content": prompt}
            ]
            start_thinking(f"Executing LLM task: {current_task.description[:30]}...")
            response = litellm.completion(model=model, messages=messages, api_key=api_key, temperature=0.3)
            stop_thinking()
            
            # Capture token usage for llm_task
            if hasattr(response, 'usage') and response.usage:
                try:
                    cost = litellm.completion_cost(completion_response=response)
                    cost_value = float(cost) if cost is not None else 0.0
                except Exception:
                    cost_value = 0.0
                
                token_usage = TokenUsage(
                    input_tokens=getattr(response.usage, 'prompt_tokens', 0),
                    output_tokens=getattr(response.usage, 'completion_tokens', 0),
                    total_tokens=getattr(response.usage, 'total_tokens', 0),
                    model=model,
                    cost=cost_value
                )
                if token_stats:
                    token_stats.add_usage(token_usage)

            llm_result = response.choices[0].message.content.strip()
            result = (f"{current_task.id}_output", {"response": llm_result})
        else:
            # Execute a regular tool using the ToolExecutor
            tool_executor = ToolExecutor()
            result = tool_executor.execute(
                tool_func=tools[tool_name],
                tool_name=tool_name,
                agent_state=state_machine.state.collected_data,
                llm_args=resolved_args,
            )

        # Process the result
        if result:
            state_machine.complete_current_task(result)
            success_memory = MemoryItem(content=f"Successfully executed {current_task.description}", type="execution_success")
            context_manager.store_memories([success_memory], "task_execution")
            context_manager.store_execution_result(state_machine.state, "execute", result, True)
            return TaskBasedExecuteResponse(success=True, result=result, reasoning=f"Successfully executed {current_task.description}", memories=[success_memory], token_usage=token_usage)
        else:
            # Handle tools that return None (fire-and-forget)
            state_machine.complete_current_task(None)
            return TaskBasedExecuteResponse(success=True, result=None, reasoning=f"Successfully executed fire-and-forget tool for task: {current_task.description}", token_usage=token_usage)

    except Exception as e:
        # Handle any failure during execution
        failure_reason = f"Tool '{tool_name}' failed during execution for task '{current_task.description}': {str(e)}"
        state_machine.fail_current_task(failure_reason)
        failure_memory = MemoryItem(content=failure_reason, type="execution_failure")
        context_manager.store_memories([failure_memory], "task_execution")
        context_manager.store_execution_result(state_machine.state, "execute", str(e), False)
        return TaskBasedExecuteResponse(success=False, result=None, reasoning=f"Tool execution failed: {str(e)}", failure_reason=failure_reason, memories=[failure_memory])


def task_based_evaluate_action(
    state_machine: TaskBasedStateMachine,
    context_manager: EnhancedContextManager,
    prompt_builder: PromptBuilder,
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    verbose: bool = False,
    token_stats: Optional[TokenStats] = None
) -> Optional[TaskBasedEvaluateResponse]:
    """
    Evaluate goal achievement based on completed tasks.
    
    Args:
        state_machine: Task-based state machine
        context_manager: RAG context manager
        prompt_builder: PromptBuilder instance
        model: LLM model to use
        api_key: API key for LLM
        verbose: Whether to show debug output
        
    Returns:
        Evaluation response with goal achievement assessment
    """
    state_machine.set_phase(AgentPhase.EVALUATING)
    
    # Build prompt
    prompt = prompt_builder.build_evaluation_prompt(state_machine, context_manager)
    
    if verbose:
        print(f"[TASK_EVALUATE] Generated prompt: {prompt[:200]}...")
    
    try:
        # Query LLM
        start_thinking("Evaluating goal achievement...")
        response = query_llm_for_model(
            prompt=prompt,
            model=model,
            output_model=TaskBasedEvaluateResponse,
            api_key=api_key,
            verbose=verbose
        )
        stop_thinking()
        
        # Capture token usage and log it dimmed
        if hasattr(query_llm_for_model, '_last_token_usage') and query_llm_for_model._last_token_usage:
            if token_stats:
                token_stats.add_usage(query_llm_for_model._last_token_usage)
                print_feedback_dimmed("TOKENS", token_stats.format_dimmed_summary(model))
        
        # Update state machine based on evaluation
        if response.goal_achieved:
            state_machine.mark_completed(True)
        else:
            # If goal is not achieved, return to planning if cycles are available
            if state_machine.planning_cycles < state_machine.max_planning_cycles:
                state_machine.start_new_planning_cycle()
            else:
                state_machine.mark_completed(False)
        
        # Store memories
        if response.memories:
            context_manager.store_memories(response.memories, "task_evaluation")
        
        # Store execution result
        context_manager.store_execution_result(state_machine.state, "evaluate", response, response.goal_achieved)
        
        return response
        
    except Exception as e:
        if verbose:
            print(f"[TASK_EVALUATE] Error: {e}")
        
        # Store failure
        context_manager.store_execution_result(state_machine.state, "evaluate", str(e), False)
        state_machine.mark_completed(False)
        
        return None


def task_based_finalize_action(
    state_machine: TaskBasedStateMachine,
    context_manager: EnhancedContextManager,
    prompt_builder: PromptBuilder,
    output_format: Optional[Type[BaseModel]] = None,
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    verbose: bool = False,
    token_stats: Optional[TokenStats] = None
) -> Optional[BaseModel]:
    """
    Finalize the task-based execution with comprehensive output.
    
    Args:
        state_machine: Task-based state machine
        context_manager: RAG context manager
        prompt_builder: PromptBuilder instance
        output_format: Desired output format
        model: LLM model to use
        api_key: API key for LLM
        verbose: Whether to show debug output
        
    Returns:
        Formatted final output
    """
    # Build comprehensive prompt
    prompt = prompt_builder.build_finalize_prompt(state_machine, context_manager, output_format)
    
    if verbose:
        memory_summary = context_manager.get_memory_summary_for_finalize()
        print(f"[TASK_FINALIZE] Using memories: {memory_summary['total_memories']}")
        print(f"[TASK_FINALIZE] Generated prompt: {prompt[:300]}...")
    
    try:
        # Query LLM
        start_thinking("Creating final output...")
        if output_format:
            response = query_llm_for_model(
                prompt=prompt,
                model=model,
                output_model=output_format,
                api_key=api_key,
                verbose=verbose
            )
        else:
            # Default response format
            class DefaultFinalOutput(BaseModel):
                result: str = Field(description="The main result/answer for the user's goal")
                summary: str = Field(description="Summary of what was accomplished")
                achievements: List[str] = Field(description="List of achievements")
                challenges: List[str] = Field(description="List of challenges encountered")
                data_collected: Dict[str, Any] = Field(description="Data collected during execution")
                
            response = query_llm_for_model(
                prompt=prompt,
                model=model,
                output_model=DefaultFinalOutput,
                api_key=api_key,
                verbose=verbose
            )
        stop_thinking()
        
        # Capture token usage and log it dimmed
        if hasattr(query_llm_for_model, '_last_token_usage') and query_llm_for_model._last_token_usage:
            if token_stats:
                token_stats.add_usage(query_llm_for_model._last_token_usage)
                print_feedback_dimmed("TOKENS", token_stats.format_dimmed_summary(model))
        
        # Store finalization memory
        finalization_memory = MemoryItem(
            content=f"Successfully finalized task-based execution for goal: {state_machine.state.goal}",
            type="completion",
            relevance="task_finalization"
        )
        context_manager.store_memories([finalization_memory], "task_finalization")
        
        return response
        
    except Exception as e:
        if verbose:
            print(f"[TASK_FINALIZE] Error: {e}")
        
        # Store failure
        failure_memory = MemoryItem(
            content=f"Failed to finalize task-based execution: {str(e)}",
            type="error",
            relevance="task_finalization_failure"
        )
        context_manager.store_memories([failure_memory], "task_finalization")
        
        return None

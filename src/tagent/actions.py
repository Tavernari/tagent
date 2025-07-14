"""
Agent actions for TAgent (plan, execute, evaluate, summarize, format).
"""

from typing import Dict, Any, Optional, Tuple, Callable, List, Type
from pydantic import BaseModel

from .llm_client import query_llm, query_llm_for_model
from .ui import print_retro_status, print_feedback_dimmed, start_thinking, stop_thinking
from .model_config import (
    get_summarizer_model,
    get_evaluator_model,
    get_planner_model,
    get_finalizer_model,
    AgentModelConfig,
)


def auto_summarize_for_evaluation(
    state: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]],
    model: str,
    api_key: Optional[str],
    verbose: bool = False,
    config: Optional[AgentModelConfig] = None,
) -> Optional[str]:
    """
    Creates a summary of collected data specifically for goal evaluation.
    This is automatically triggered when there's sufficient data but evaluation needs clarity.
    """
    goal = state.get("goal", "")
    
    # Extract relevant conversation data
    tool_results = []
    if conversation_history:
        for message in conversation_history:
            if message.get('role') == 'user' and 'Observation' in message.get('content', ''):
                tool_results.append(message.get('content', ''))
    
    if len(tool_results) < 2:  # Not enough data to summarize
        return None
    
    prompt = (
        f"Goal: '{goal}'\n\n"
        f"Current State: {state}\n\n"
        f"Tool Execution Results:\n" + "\n".join(tool_results[-10:]) + "\n\n"  # Last 10 results
        "Create a concise summary focusing on:\n"
        "1. What data has been successfully collected\n"
        "2. Key findings and results from tool executions\n"
        "3. Whether sufficient information exists to achieve the goal\n"
        "4. Any missing pieces needed for goal completion\n\n"
        "This summary will be used for goal evaluation."
    )
    
    start_thinking("Auto-summarizing for evaluation")
    try:
        summarizer_model = get_summarizer_model(model, config)
        response = query_llm(
            prompt,
            summarizer_model,
            api_key,
            verbose=verbose,
        )
        if response and response.reasoning:
            return response.reasoning
        return None
    finally:
        stop_thinking()


def enhanced_goal_evaluation_action(
    state: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    tools: Optional[Dict[str, Callable]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    verbose: bool = False,
    store: Optional[Any] = None,
    auto_summarize: bool = True,
    config: Optional[AgentModelConfig] = None,
) -> Optional[Tuple[str, BaseModel]]:
    """
    Enhanced evaluator with optional auto-summarization before evaluation.
    """
    # Check if auto-summarization would be helpful
    if auto_summarize and conversation_history:
        tool_observations = [
            msg for msg in conversation_history 
            if msg.get('role') == 'user' and 'Observation' in msg.get('content', '')
        ]
        
        # Trigger auto-summarization if there are many observations but no explicit summary
        if len(tool_observations) >= 3 and not state.get("summary"):
            if verbose:
                print("[AUTO-SUMMARIZE] Triggering auto-summarization for better evaluation...")
            
            auto_summary = auto_summarize_for_evaluation(
                state, conversation_history, model, api_key, verbose, config
            )
            if auto_summary:
                # Add the summary to state temporarily for this evaluation
                state = state.copy()  # Don't modify original state
                state["auto_summary"] = auto_summary
                if verbose:
                    print(f"[AUTO-SUMMARIZE] Generated summary added to evaluation context")
    
    # Use the original evaluation function with enhanced context
    return goal_evaluation_action(
        state, model, api_key, tools, conversation_history, verbose, store, config
    )


def plan_action(
    state: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    tools: Optional[Dict[str, Callable]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    verbose: bool = False,
    config: Optional[AgentModelConfig] = None,
) -> Optional[Tuple[str, BaseModel]]:
    """Generates a plan via LLM structured output, adapting to evaluator feedback."""
    print_retro_status("PLAN", "Analyzing current situation...")
    goal = state.get("goal", "")
    used_tools = state.get("used_tools", [])
    available_tools = list(tools.keys()) if tools else []
    unused_tools = [t for t in available_tools if t not in used_tools]

    print_retro_status(
        "PLAN", f"Tools used: {len(used_tools)}, Unused: {len(unused_tools)}"
    )

    # Extract feedback to adapt the prompt
    evaluation_result = state.get("evaluation_result", {})
    feedback = evaluation_result.get("feedback", "")
    missing_items = evaluation_result.get("missing_items", [])
    suggestions = evaluation_result.get("suggestions", [])

    feedback_str = ""
    if feedback or missing_items or suggestions:
        feedback_str = (
            f"\nPrevious Evaluator Feedback: {feedback}\n"
            f"Missing Items: {missing_items}\n"
            f"Suggestions: {suggestions}\n"
            "Address this feedback in your new plan. Incorporate suggestions, "
            "focus on missing items, and use unused tools where appropriate."
        )

    prompt = (
        f"Goal: {goal}\n"
        f"Current progress: {state}\n"
        f"Used tools: {used_tools}\n"
        f"Unused tools: {unused_tools}\n"
        f"{feedback_str}\n"
        "The current approach may not be working. Generate a new strategic plan. "
        "Consider: 1) What data is still missing? 2) What tools haven't been tried? "
        "3) What alternative approaches could work? 4) Should we try different "
        "parameters? Output a plan as params (e.g., {'steps': ['step1', 'step2'], "
        "'focus_tools': ['tool1']})."
    )
    start_thinking("Generating strategic plan")
    try:
        planner_model = get_planner_model(model, config)
        response = query_llm(
            prompt,
            planner_model,
            api_key,
            tools=tools,
            conversation_history=conversation_history,
            verbose=verbose,
        )
        # Validate response action type
        if response.action != "plan":
            if verbose:
                print(
                    f"[WARNING] Unexpected action in plan: {response.action}. "
                    "Requesting plan-specific response."
                )
            clarified_prompt = (
                prompt
                + "\nPlease provide a strategic plan with action='plan' and appropriate params."
            )
            response = query_llm(
                clarified_prompt,
                planner_model,
                api_key,
                tools=tools,
                conversation_history=conversation_history,
                verbose=verbose,
            )
        if response.action == "plan":
            plan_params = response.params
            print_retro_status("SUCCESS", "Strategic plan generated")

            # Show plan feedback in non-verbose mode
            if not verbose and response.reasoning:
                print_feedback_dimmed("PLAN_FEEDBACK", response.reasoning)

            return (
                "plan",
                plan_params,
            )
        else:
            if verbose:
                print("[ERROR] Failed to get valid 'plan' response after retry.")
            return None
    finally:
        stop_thinking()


def summarize_action(
    state: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    tools: Optional[Dict[str, Callable]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    verbose: bool = False,
    config: Optional[AgentModelConfig] = None,
) -> Optional[Tuple[str, BaseModel]]:
    """Summarizes the context, adapting to evaluator feedback."""
    print_retro_status("SUMMARIZE", "Compiling collected information...")

    # Extract feedback if available
    evaluation_result = state.get("evaluation_result", {})
    feedback = evaluation_result.get("feedback", "")
    missing_items = evaluation_result.get("missing_items", [])
    suggestions = evaluation_result.get("suggestions", [])

    feedback_str = ""
    if feedback or missing_items or suggestions:
        feedback_str = (
            f"\nPrevious Evaluator Feedback: {feedback}\nMissing: {missing_items}\n"
            f"Suggestions: {suggestions}\nIncorporate this feedback into the summary. "
            "Ensure all suggestions are addressed."
        )

    prompt = (
        f"Based on the current state: {state}. Generate a detailed summary that "
        f"fulfills the goal.{feedback_str}"
    )
    start_thinking("Compiling summary")
    try:
        summarizer_model = get_summarizer_model(model, config)
        response = query_llm(
            prompt,
            summarizer_model,
            api_key,
            tools=tools,
            conversation_history=conversation_history,
            verbose=verbose,
        )
        if response.action != "summarize":
            if verbose:
                print(
                    f"[WARNING] Invalid action in summarize: {response.action}. "
                    "Requesting summary-specific response."
                )
            clarified_prompt = prompt + "\nPlease provide a summary with action='summarize'."
            response = query_llm(
                clarified_prompt,
                summarizer_model,
                api_key,
                tools=tools,
                conversation_history=conversation_history,
                verbose=verbose,
            )
        if response.action == "summarize":
            summary_content = response.params.get("content") or response.reasoning
            summary = {
                "content": summary_content,
                "calculated_from_feedback": bool(feedback_str),
            }
            print_retro_status("SUCCESS", "Summary generated successfully")

            # Show summary feedback in non-verbose mode
            if not verbose and response.reasoning:
                print_feedback_dimmed("FEEDBACK", response.reasoning)

            return ("summary", summary)
    finally:
        stop_thinking()
    return None


def goal_evaluation_action(
    state: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    tools: Optional[Dict[str, Callable]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    verbose: bool = False,
    store: Optional[
        Any
    ] = None,  # Store reference for conversation updates (legacy bug fix)
    config: Optional[AgentModelConfig] = None,
) -> Optional[Tuple[str, BaseModel]]:
    """
    Evaluates if the goal has been achieved via structured output, 
    considering both state data and conversation history.
    """
    print_retro_status("EVALUATE", "Checking if goal was achieved...")
    goal = state.get("goal", "")
    
    # Analyze both state and conversation history for a complete picture
    data_items = [
        k for k, v in state.items() if k not in ["goal", "achieved", "used_tools"] and v
    ]
    
    # Extract tool results from conversation history
    conversation_data = []
    if conversation_history:
        for message in conversation_history:
            if message.get('role') == 'user' and 'Observation' in message.get('content', ''):
                conversation_data.append(message.get('content', ''))
    
    print_retro_status("EVALUATE", f"Analyzing {len(data_items)} state items + {len(conversation_data)} conversation observations")

    # Create comprehensive context including conversation history
    conversation_context = ""
    if conversation_data:
        conversation_context = (
            "\n\nTool Execution Results from Conversation History:\n" +
            "\n".join(conversation_data[-5:])  # Last 5 observations to avoid token limit
        )
    
    # Include auto-generated summary if available
    auto_summary_context = ""
    if state.get("auto_summary"):
        auto_summary_context = f"\n\nData Summary for Evaluation:\n{state['auto_summary']}\n"

    # Extract previous feedback for context
    evaluation_result = state.get("evaluation_result", {})
    previous_feedback = evaluation_result.get("feedback", "")
    previous_missing = evaluation_result.get("missing_items", [])

    feedback_str = ""
    if previous_feedback or previous_missing:
        feedback_str = (
            f"\nPrevious Evaluation: {previous_feedback}\n"
            f"Previously Missing: {previous_missing}\n"
            "Consider if these have been addressed in the current state or conversation history. "
            "Be consistent with past evaluations."
        )

    prompt = (
        f"Goal: '{goal}'\n\n"
        f"Current State Data: {state}\n"
        f"{conversation_context}\n"
        f"{auto_summary_context}\n"
        f"{feedback_str}\n\n"
        "EVALUATION INSTRUCTIONS:\n"
        "1. Review ALL available information: state data, conversation history, and any data summary\n"
        "2. Check if all required information to achieve the goal has been collected\n"
        "3. Consider if the data is sufficient for comparison, calculation, or analysis as needed\n"
        "4. If the goal involves comparing items, ensure both items' data is available\n"
        "5. Pay special attention to the data summary which consolidates collected information\n"
        "6. If NOT achieved, be specific about what data is missing or insufficient\n\n"
        "Output: If goal achieved, set 'achieved': true. If not, explain missing items and provide suggestions."
    )
    start_thinking("Evaluating goal")
    try:
        evaluator_model = get_evaluator_model(model, config)
        response = query_llm(
            prompt,
            evaluator_model,
            api_key,
            tools=tools,
            conversation_history=conversation_history,
            verbose=verbose,
        )
        # Validate response action type
        if response.action != "evaluate":
            if verbose:
                print(
                    f"[WARNING] Unexpected action in evaluate: {response.action}. "
                    "Requesting evaluation-specific response."
                )
            clarified_prompt = (
                prompt
                + "\nPlease provide an evaluation with action='evaluate' and params containing "
                "'achieved' (bool), and if not achieved, 'missing_items' and 'suggestions'."
            )
            response = query_llm(
                clarified_prompt,
                evaluator_model,
                api_key,
                tools=tools,
                conversation_history=conversation_history,
                verbose=verbose,
            )
        if response.action == "evaluate":
            achieved = bool(response.params.get("achieved", False))
            evaluation_feedback = response.reasoning
            if achieved:
                print_retro_status("SUCCESS", "✓ Goal was achieved!")
                return ("achieved", achieved)
            else:
                print_retro_status("INFO", "✗ Goal not yet achieved")

                # Show evaluation feedback in non-verbose mode
                if not verbose:
                    if evaluation_feedback:
                        print_feedback_dimmed("FEEDBACK", evaluation_feedback)
                    missing_items = response.params.get("missing_items", [])
                    if missing_items:
                        missing_strings = [
                            str(item) if not isinstance(item, str) else item
                            for item in missing_items
                        ]
                        print_feedback_dimmed("MISSING", ", ".join(missing_strings))
                    suggestions = response.params.get("suggestions", [])
                    if suggestions:
                        print_feedback_dimmed("SUGGESTIONS", ", ".join(suggestions))

                evaluation_dict = {
                    "achieved": achieved,
                    "feedback": evaluation_feedback,
                    "missing_items": response.params.get("missing_items", []),
                    "suggestions": response.params.get("suggestions", []),
                }

                # Add observation to history immediately after failure
                # Note: store parameter was added to fix a bug where store was
                # referenced but not passed
                if store is not None:
                    missing_str = ", ".join(
                        response.params.get("missing_items", [])
                    )
                    suggestions_str = ", ".join(
                        response.params.get("suggestions", [])
                    )
                    observation = (
                        f"Observation from evaluate: Goal NOT achieved. "
                        f"Feedback: {evaluation_feedback}. Missing: {missing_str}. Suggestions: {suggestions_str}. "
                        "Consider planning or executing next actions to address this feedback."
                    )
                    store.add_to_conversation("user", observation)

                return ("evaluation_result", evaluation_dict)
        else:
            if verbose:
                print("[ERROR] Failed to get valid 'evaluate' response after retry.")
            return None
    finally:
        stop_thinking()


def format_output_action(
    state: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    output_format: Type[BaseModel],
    verbose: bool = False,
    config: Optional[AgentModelConfig] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Tuple[str, BaseModel]]:
    """Formats the final output according to the specified Pydantic model."""
    print_retro_status("FORMAT", "Structuring final result...")
    goal = state.get("goal", "")

    # Add conversation history to the prompt for better context
    conversation_summary = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in conversation_history]
    ) if conversation_history else "No conversation history available."

    prompt = (
        f"Based on the final state: {state}, the original goal: '{goal}', and the conversation history, "
        f"please structure the final result. Conversation History:\n{conversation_summary}\n\n"
        "Extract and format all relevant data collected during the goal "
        "execution. Create appropriate summaries and ensure all required "
        "fields are filled according to the output schema."
    )
    start_thinking("Structuring final result")
    try:
        finalizer_model = get_finalizer_model(model, config)
        formatted_output = query_llm_for_model(
            prompt,
            finalizer_model,
            output_format,
            api_key,
            verbose=verbose,
            conversation_history=conversation_history,
        )
    finally:
        stop_thinking()
    print_retro_status("SUCCESS", "Result structured successfully")
    return ("final_output", formatted_output)


def format_fallback_output_action(
    state: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    output_format: Type[BaseModel],
    verbose: bool = False,
    config: Optional[AgentModelConfig] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Tuple[str, BaseModel]]:
    """
    Formats output with fallback handling for incomplete data.
    
    This function is designed to work even when the goal hasn't been fully achieved
    or when max iterations are reached, ensuring the client always gets a structured
    response according to the output schema.
    """
    print_retro_status("FORMAT_FALLBACK", "Structuring available data...")
    goal = state.get("goal", "")

    conversation_summary = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in conversation_history]
    ) if conversation_history else "No conversation history available."
    
    prompt = (
        f"Based on the current state: {state} and the original goal: '{goal}'. "
        f"Here is the conversation history that led to this state:\n{conversation_summary}\n\n"
        "Note: The goal may not be fully achieved and data may be incomplete. "
        "Please extract and format available data collected so far according to the output schema. "
        "For missing required fields, provide reasonable defaults or indicate unavailability "
        "(e.g., 'Data not available', 'Not collected', etc.). "
        "Please fill required schema fields with the best available information. "
        "Consider creating meaningful summaries based on the data that was successfully gathered."
    )
    
    start_thinking("Structuring available data with fallback")
    try:
        finalizer_model = get_finalizer_model(model, config)
        formatted_output = query_llm_for_model(
            prompt,
            finalizer_model,
            output_format,
            api_key,
            verbose=verbose,
            conversation_history=conversation_history,
        )
    finally:
        stop_thinking()
    print_retro_status("SUCCESS", "Fallback result structured successfully")
    return ("final_output", formatted_output)


def finalize_action(
    state: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    output_format: Optional[Type[BaseModel]],
    verbose: bool = False,
    config: Optional[AgentModelConfig] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Tuple[str, BaseModel]]:
    """Finalizes the output: structures via schema if provided, else free-form LLM summary."""
    print_retro_status("FINALIZE", "Generating final response...")

    if output_format:
        # Use structured formatting
        try:
            formatted_output = format_output_action(
                state,
                model,
                api_key,
                output_format,
                verbose=verbose,
                config=config,
                conversation_history=conversation_history,
            )
            state["final_output"] = formatted_output[1]  # Update state
            print_retro_status("SUCCESS", "Output formatted successfully")
            return ("final_output", formatted_output[1])
        except Exception as e:
            print_retro_status("ERROR", f"Formatting failed: {str(e)}")
            # Fallback to free-form if formatting fails
            pass

    # Free-form LLM-generated result if no schema or formatting failed
    conversation_summary = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in conversation_history]
    ) if conversation_history else "No conversation history available."
    prompt = (
        f"Goal: {state.get('goal', '')}\n"
        f"Current state: {state}\n"
        f"Conversation History:\n{conversation_summary}\n\n"
        "Generate a final summary or result based on all collected data."
    )
    start_thinking("Generating final result")
    try:
        finalizer_model = get_finalizer_model(model, config)
        response = query_llm(
            prompt,
            finalizer_model,
            api_key,
            verbose=verbose,
            conversation_history=conversation_history,
        )
        final_result = response.params.get("content") or response.reasoning
        state["final_output"] = final_result
        print_retro_status("SUCCESS", "Free-form result generated")
        return ("final_output", final_result)
    finally:
        stop_thinking()

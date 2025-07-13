"""
Agent actions for TAgent (plan, execute, evaluate, summarize, format).
"""

from typing import Dict, Any, Optional, Tuple, Callable, List, Type
from pydantic import BaseModel

from .llm_client import query_llm, query_llm_for_model
from .ui import print_retro_status, print_feedback_dimmed, start_thinking, stop_thinking


def plan_action(
    state: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    tools: Optional[Dict[str, Callable]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    verbose: bool = False,
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
            "Address this feedback in your new plan. Incorporate suggestions, focus on missing items, and use unused tools where appropriate."
        )

    prompt = (
        f"Goal: {goal}\n"
        f"Current progress: {state}\n"
        f"Used tools: {used_tools}\n"
        f"Unused tools: {unused_tools}\n"
        f"{feedback_str}\n"
        "The current approach may not be working. Generate a new strategic plan. "
        "Consider: 1) What data is still missing? 2) What tools haven't been tried? "
        "3) What alternative approaches could work? 4) Should we try different parameters? "
        "Output a plan as params (e.g., {'steps': ['step1', 'step2'], 'focus_tools': ['tool1']})."
    )
    start_thinking("Generating strategic plan")
    try:
        response = query_llm(
            prompt,
            model,
            api_key,
            tools=tools,
            conversation_history=conversation_history,
            verbose=verbose,
        )
        # Validation: Force action='plan' if wrong
        if response.action != "plan":
            if verbose:
                print(
                    f"[WARNING] Invalid action in plan: {response.action}. Retrying with forced plan."
                )
            forced_prompt = (
                prompt
                + "\nMUST use action='plan' and provide params with a strategic plan."
            )
            response = query_llm(
                forced_prompt,
                model,
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
        feedback_str = f"\nPrevious Evaluator Feedback: {feedback}\nMissing: {missing_items}\nSuggestions: {suggestions}\nIncorporate this feedback into the summary. Ensure all suggestions are addressed."

    prompt = (
        f"Based on the current state: {state}. Generate a detailed summary that fulfills the goal.{feedback_str}\n"
        "Include calculations (e.g., total estimated cost based on stock and prices), order predictions, and analysis. Make it comprehensive."
    )
    start_thinking("Compiling summary")
    try:
        response = query_llm(
            prompt,
            model,
            api_key,
            tools=tools,
            conversation_history=conversation_history,
            verbose=verbose,
        )
        if response.action != "summarize":
            if verbose:
                print(
                    f"[WARNING] Invalid action in summarize: {response.action}. Retrying with forced summarize."
                )
            forced_prompt = prompt + "\nMUST use action='summarize'."
            response = query_llm(
                forced_prompt,
                model,
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
    ] = None,  # Store reference for conversation updates (Optional to fix legacy bug)
) -> Optional[Tuple[str, BaseModel]]:
    """Evaluates if the goal has been achieved via structured output, considering previous feedback."""
    print_retro_status("EVALUATE", "Checking if goal was achieved...")
    goal = state.get("goal", "")
    data_items = [
        k for k, v in state.items() if k not in ["goal", "achieved", "used_tools"] and v
    ]
    print_retro_status("EVALUATE", f"Analyzing {len(data_items)} collected data items")

    # Extract previous feedback for context
    evaluation_result = state.get("evaluation_result", {})
    previous_feedback = evaluation_result.get("feedback", "")
    previous_missing = evaluation_result.get("missing_items", [])

    feedback_str = ""
    if previous_feedback or previous_missing:
        feedback_str = (
            f"\nPrevious Evaluation: {previous_feedback}\n"
            f"Previously Missing: {previous_missing}\n"
            "Consider if these have been addressed in the current state. Be consistent with past evaluations."
        )

    prompt = (
        f"Based on the current state: {state} and the goal: '{goal}'.{feedback_str}\n"
        "Evaluate if the goal has been sufficiently achieved. Consider the data collected and whether it meets the requirements. "
        "If NOT achieved, explain specifically what is missing or insufficient in 'reasoning', and ALWAYS include 'missing_items' (list of strings) and 'suggestions' (list of at least 2 specific actions) in params so the agent can take corrective action."
    )
    start_thinking("Evaluating goal")
    try:
        response = query_llm(
            prompt,
            model,
            api_key,
            tools=tools,
            conversation_history=conversation_history,
            verbose=verbose,
        )
        # Validation: Force action='evaluate' if wrong
        if response.action != "evaluate":
            if verbose:
                print(
                    f"[WARNING] Invalid action in evaluate: {response.action}. Retrying with forced evaluate."
                )
            forced_prompt = (
                prompt
                + "\nMUST use action='evaluate' and provide params with 'achieved' (bool), 'missing_items', 'suggestions' if not achieved."
            )
            response = query_llm(
                forced_prompt,
                model,
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
                # Note: store parameter was added to fix a bug where store was referenced but not passed
                if store is not None:
                    missing_str = ", ".join(response.params.get("missing_items", []))
                    suggestions_str = ", ".join(response.params.get("suggestions", []))
                    observation = (
                        f"Observation from evaluate: Goal NOT achieved. "
                        f"Feedback: {evaluation_feedback}. Missing: {missing_str}. Suggestions: {suggestions_str}. "
                        "MUST plan or execute next to address this."
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
) -> Optional[Tuple[str, BaseModel]]:
    """Formats the final output according to the specified Pydantic model."""
    print_retro_status("FORMAT", "Structuring final result...")
    goal = state.get("goal", "")
    prompt = (
        f"Based on the final state: {state} and the original goal: '{goal}'. "
        "Extract and format all relevant data collected during the goal execution. "
        "Create appropriate summaries and ensure all required fields are filled according to the output schema."
    )
    start_thinking("Structuring final result")
    try:
        formatted_output = query_llm_for_model(
            prompt, model, output_format, api_key, verbose=verbose
        )
    finally:
        stop_thinking()
    print_retro_status("SUCCESS", "Result structured successfully")
    return ("final_output", formatted_output)

# TAgent main module - orchestrates the agent execution loop.
# Integration with LiteLLM for real LLM calls, leveraging JSON Mode.
# Requirements: pip install pydantic litellm

from typing import Dict, Any, Optional, Callable, Type
from pydantic import BaseModel, Field
import json
import time

from .version import __version__
from .store import Store
from .llm_client import query_llm, generate_step_title
from .actions import plan_action, summarize_action, goal_evaluation_action, format_output_action
from .ui import (
    print_retro_banner, print_retro_status, print_retro_step, print_feedback_dimmed,
    start_thinking, stop_thinking, Colors
)
from .utils import detect_action_loop, format_conversation_as_chat


# === Main Agent Loop ===
def run_agent(
    goal: str,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    max_iterations: int = 20,
    tools: Optional[Dict[str, Callable]] = None,
    output_format: Optional[Type[BaseModel]] = None,
    verbose: bool = False,
    crash_if_over_iterations: bool = False,
) -> Optional[BaseModel]:
    """
    Runs the main agent loop.

    Args:
        goal: The main objective for the agent.
        model: The LLM model to use.
        api_key: The API key for the LLM service.
        max_iterations: The maximum number of iterations.
        tools: A dictionary of custom tools to register with the agent.
        output_format: The Pydantic model for the final output.
        verbose: If True, shows all debug logs. If False, shows only essential logs.
        crash_if_over_iterations: If True, raises exception when max_iterations reached.
                                 If False (default), returns results with summarizer fallback.

    Returns:
        An instance of the `output_format` model, or None if no output is generated.
    """
    # 90s Style Agent Initialization
    print_retro_banner(f"T-AGENT v{__version__} STARTING", "▓", color=Colors.BRIGHT_MAGENTA)
    print_retro_status("INIT", f"Goal: {goal[:40]}...")
    print_retro_status("CONFIG", f"Model: {model} | Max Iterations: {max_iterations}")

    store = Store({"goal": goal, "results": [], "used_tools": []})

    # Infinite loop protection system
    consecutive_failures = 0
    max_consecutive_failures = 5
    last_data_count = 0

    # Action loop detection system
    recent_actions = []
    max_recent_actions = 3
    
    # Step counting and evaluator tracking
    evaluation_rejections = 0
    max_evaluation_rejections = 2

    # Register tools if provided
    if tools:
        print_retro_status("TOOLS", f"Registering {len(tools)} tools...")
        for name, tool_func in tools.items():
            store.register_tool(name, tool_func)
            print_retro_status("TOOL_REG", f"[{name}] loaded successfully")

    print_retro_banner("STARTING MAIN LOOP", "~", color=Colors.BRIGHT_GREEN)
    iteration = 0
    while (
        not store.state.data.get("achieved", False)
        and iteration < max_iterations
        and consecutive_failures < max_consecutive_failures
    ):
        iteration += 1

        # Add step counting warnings
        remaining_steps = max_iterations - iteration
        if remaining_steps <= 3:
            print_retro_status(
                "WARNING", 
                f"Only {remaining_steps} steps remaining before reaching max iterations ({max_iterations})"
            )
            
        if verbose:
            print(f"[LOOP] Iteration {iteration}/{max_iterations}. Current state: {store.state.data}")
        else:
            print_retro_status("STEP", f"Step {iteration}/{max_iterations}")

        # Check if there was real progress (reset failure counter)
        data_keys = [
            k
            for k, v in store.state.data.items()
            if k not in ["goal", "achieved", "used_tools"] and v
        ]
        current_data_count = len(data_keys)

        if current_data_count > last_data_count:
            if verbose:
                print(
                    f"[PROGRESS] Data items increased from {last_data_count} to {current_data_count} - resetting failure counter"
                )
            consecutive_failures = 0
            last_data_count = current_data_count

        progress_summary = f"Progress: {current_data_count} data items collected"

        used_tools = store.state.data.get("used_tools", [])
        unused_tools = [t for t in store.tools.keys() if t not in used_tools]

        # Check if the last action was 'evaluate' failure
        last_action_was_failed_evaluate = (
            recent_actions and recent_actions[-1] == "evaluate"
            and not store.state.data.get("achieved", False)
        )

        # Detect action loop and adjust strategy
        action_loop_detected = detect_action_loop(recent_actions, max_recent_actions)
        strategy_hint = ""

        if action_loop_detected:
            last_action = recent_actions[-1] if recent_actions else "unknown"
            if verbose:
                print_retro_status(
                    "WARNING", f"Loop detected: repeating '{last_action}'"
                )
                print(
                    f"[STRATEGY] Action loop detected: repeating '{last_action}' - suggesting strategy change"
                )

            if last_action == "evaluate" and unused_tools:
                strategy_hint = f"CRITICAL: Stop evaluating! The goal is NOT achieved because you need to gather more data. Use unused tools: {unused_tools}. DO NOT use 'evaluate' again. "
            elif last_action == "evaluate" and not unused_tools:
                strategy_hint = "CRITICAL: Stop evaluating! The goal is NOT achieved. You must 'plan' a new strategy or 'execute' tools with different parameters. DO NOT use 'evaluate' again until you've tried other actions. "
            elif unused_tools:
                strategy_hint = f"IMPORTANT: Break the pattern! Try unused tools: {unused_tools} or use 'plan' to rethink approach. "
            else:
                strategy_hint = "IMPORTANT: Break the pattern! Try 'plan' to develop new strategy or different parameters. "

        # Include evaluation feedback in prompt if available
        evaluation_feedback = ""
        evaluation_result = store.state.data.get("evaluation_result", {})
        if evaluation_result and not store.state.data.get("achieved", False):
            feedback = evaluation_result.get("feedback", "")
            missing_items = evaluation_result.get("missing_items", [])
            suggestions = evaluation_result.get("suggestions", [])

            if feedback or missing_items or suggestions:
                evaluation_feedback = f"\nEVALUATOR FEEDBACK: {feedback}"
                if missing_items:
                    evaluation_feedback += f"\nMISSING: {missing_items}"
                if suggestions:
                    evaluation_feedback += f"\nSUGGESTIONS: {suggestions}"
                evaluation_feedback += "\nACT ON THIS FEEDBACK TO IMPROVE THE RESULT.\n"

        # Add step count warnings to prompt
        remaining_steps = max_iterations - iteration
        step_warning = ""
        if remaining_steps <= 5:
            if remaining_steps <= 1:
                step_warning = f"⚠️ CRITICAL: Only {remaining_steps} step remaining before agent stops! Must accomplish goal NOW or it will be lost. "
            elif remaining_steps <= 2:
                step_warning = f"⚠️ WARNING: Only {remaining_steps} steps remaining! Focus on goal completion. "
            else:
                step_warning = f"⚠️ {remaining_steps} steps left. Be efficient. "
        
        # Add instruction to avoid evaluate after failure
        avoid_evaluate_hint = ""
        if last_action_was_failed_evaluate:
            avoid_evaluate_hint = (
                "CRITICAL: The last evaluation failed. DO NOT choose 'evaluate' again! "
                "You MUST choose 'plan' to rethink strategy based on feedback, or 'execute' to gather more data. "
                "Address missing items and suggestions from the evaluator."
            )

        prompt = (
            f"Goal: {goal}\n"
            f"Current state: {store.state.data}\n"
            f"{progress_summary}\n"
            f"Step {iteration}/{max_iterations}. {step_warning}\n"
            f"Used tools: {used_tools}\n"
            f"Unused tools: {unused_tools}\n"
            f"{evaluation_feedback}"
            f"{strategy_hint}"
            f"{avoid_evaluate_hint}"
            "For 'execute' action, prefer UNUSED tools to gather different types of data. "
            "If goal evaluation fails, DO NOT immediately evaluate again - try other actions first. "
            "Only use 'evaluate' after making changes or gathering new data. "
            "Available actions: plan, execute, summarize, evaluate"
        )
        # Add current prompt to history
        store.add_to_conversation("user", prompt)

        print_retro_status("THINKING", "Consulting AI for next action...")
        start_thinking("Thinking")
        try:
            decision = query_llm(
                prompt,
                model,
                api_key,
                tools=store.tools,
                conversation_history=store.conversation_history[:-1],
                verbose=verbose,
            )  # Exclude last message to avoid duplication
        finally:
            stop_thinking()

        # Generate concise step title using LLM
        step_title = generate_step_title(
            decision.action, decision.reasoning, model, api_key, verbose
        )
        print_retro_step(iteration, decision.action, step_title)
        if verbose:
            print(f"[DECISION] LLM decided: {decision}")

        # Track recent actions to detect loops
        recent_actions.append(decision.action)
        if len(recent_actions) > max_recent_actions:
            recent_actions.pop(0)  # Keep only the latest actions

        # Force action if 'evaluate' after previous failure
        if decision.action == "evaluate" and last_action_was_failed_evaluate:
            print_retro_status("WARNING", "Preventing evaluate loop - forcing 'plan' instead")
            decision.action = "plan"
            decision.reasoning = "Forced plan due to evaluate loop prevention"
            # Add to history as observation
            store.add_to_conversation("user", "Observation: Evaluate loop detected. Forcing plan to address evaluator feedback.")

        # Add assistant response to history
        store.add_assistant_response(decision)

        # Dispatch based on LLM decision
        if decision.action == "plan":
            print_retro_status("PLAN", "Generating strategic plan...")
            store.dispatch(
                lambda state: plan_action(
                    state,
                    model,
                    api_key,
                    tools=store.tools,
                    conversation_history=store.conversation_history,
                    verbose=verbose,
                ),
                verbose=verbose,
            )
        elif decision.action == "execute":
            # Extract tool and args from the main decision
            tool_name = decision.params.get("tool")
            tool_args = decision.params.get("args", {})
            if tool_name and tool_name in store.tools:
                print_retro_status("EXECUTE", f"Executing tool: {tool_name}")
                result = store.tools[tool_name](store.state.data, tool_args)
                if result:
                    # Update state
                    if isinstance(result, list):
                        for item in result:
                            if isinstance(item, tuple) and len(item) == 2:
                                key, value = item
                                store.state.data[key] = value
                    elif isinstance(result, tuple) and len(result) == 2:
                        key, value = result
                        store.state.data[key] = value

                    # Track used tools
                    used_tools = store.state.data.get("used_tools", [])
                    if tool_name not in used_tools:
                        used_tools.append(tool_name)
                        store.state.data["used_tools"] = used_tools

                    # Add result to conversation history
                    if isinstance(result, list):
                        formatted_result = {
                            k: v
                            for (k, v) in result
                            if isinstance(v, (dict, list, str))
                        }
                        tool_output = json.dumps(formatted_result, indent=2)
                    elif isinstance(result, tuple) and len(result) == 2:
                        key, value = result
                        tool_output = json.dumps({key: value}, indent=2)
                    else:
                        tool_output = str(result)
                    observation = f"Observation from tool {tool_name}: {tool_output}"
                    store.add_to_conversation("user", observation)

                    print_retro_status(
                        "SUCCESS",
                        f"Tool {tool_name} executed successfully. Observation added to history as user message.",
                    )
                else:
                    observation = f"Observation from tool {tool_name}: Execution failed or returned no result."
                    store.add_to_conversation("user", observation)
                    print_retro_status(
                        "WARNING",
                        f"Tool {tool_name} returned no result. Observation added.",
                    )
            else:
                print_retro_status("ERROR", f"Tool not found: {tool_name}")
                observation = f"Error: Tool {tool_name} not found."
                store.add_to_conversation("user", observation)
                if verbose:
                    print(
                        f"[ERROR] Tool not found: {tool_name}. Available tools: {list(store.tools.keys())}"
                    )
        elif decision.action == "summarize":
            print_retro_status("SUMMARIZE", "Generating progress summary...")
            store.dispatch(
                lambda state: summarize_action(
                    state,
                    model,
                    api_key,
                    tools=store.tools,
                    conversation_history=store.conversation_history,
                    verbose=verbose,
                ),
                verbose=verbose,
            )
        elif decision.action == "evaluate":
            print_retro_status("EVALUATE", "Evaluating if goal was achieved...")
            # Store previous state to detect change
            previous_achieved = store.state.data.get("achieved", False)
            store.dispatch(
                lambda state: goal_evaluation_action(
                    state,
                    model,
                    api_key,
                    tools=store.tools,
                    conversation_history=store.conversation_history,
                    verbose=verbose,
                    store=store,
                ),
                verbose=verbose,
            )

            # Check evaluation result and get detailed feedback
            current_achieved = store.state.data.get("achieved", False)
            evaluation_result = store.state.data.get("evaluation_result", {})

            if not current_achieved and not previous_achieved:
                consecutive_failures += 1
                evaluation_rejections += 1

                # Extract and show specific feedback from evaluator
                feedback = evaluation_result.get(
                    "feedback", "No specific feedback provided"
                )
                missing_items = evaluation_result.get("missing_items", [])
                suggestions = evaluation_result.get("suggestions", [])

                # Show evaluator rejection message with specific reason
                if consecutive_failures == 1:
                    print_retro_status(
                        "INFO", "Evaluator rejected - working on task again"
                    )
                    if verbose and feedback:
                        print(f"[FEEDBACK] Evaluator says: {feedback}")
                    elif not verbose:
                        if feedback:
                            print_feedback_dimmed("FEEDBACK", feedback)
                        if missing_items:
                            missing_strings = [str(item) if not isinstance(item, str) else item for item in missing_items]
                            print_feedback_dimmed("MISSING", ", ".join(missing_strings))
                        if suggestions:
                            print_feedback_dimmed("SUGGESTIONS", ", ".join(suggestions))
                elif consecutive_failures <= max_consecutive_failures:
                    print_retro_status(
                        "INFO",
                        f"Evaluator rejected {consecutive_failures} times - continuing work",
                    )
                    if verbose and feedback:
                        print(f"[FEEDBACK] Evaluator says: {feedback}")
                    elif not verbose:
                        if feedback:
                            print_feedback_dimmed("FEEDBACK", feedback)
                        if missing_items:
                            missing_strings = [str(item) if not isinstance(item, str) else item for item in missing_items]
                            print_feedback_dimmed("MISSING", ", ".join(missing_strings))
                        if suggestions:
                            print_feedback_dimmed("SUGGESTIONS", ", ".join(suggestions))

                if verbose:
                    print(
                        f"[FAILURE] Evaluator failed {consecutive_failures}/{max_consecutive_failures} times consecutively"
                    )

                # Enhanced evaluator recursion prevention
                if evaluation_rejections >= max_evaluation_rejections:
                    print_retro_status(
                        "WARNING",
                        f"Evaluator rejected {evaluation_rejections} times - preventing evaluation loops",
                    )
                    # Force plan_action immediately
                    store.dispatch(
                        lambda state: plan_action(
                            state, model, api_key, tools=store.tools, conversation_history=store.conversation_history, verbose=verbose
                        ),
                        verbose=verbose,
                    )
                    recent_actions.append("plan")  # Update to break loop
                
                # After 2 evaluation failures, force alternative actions
                if consecutive_failures >= 2:
                    if verbose:
                        print_retro_status(
                            "WARNING",
                            "Too many evaluation failures - forcing strategy change",
                        )
                    # Skip the next evaluate decision by manipulating recent actions
                    recent_actions.append(
                        "evaluate"
                    )  # Add extra evaluate to trigger loop detection

                # Force completion if many consecutive failures with sufficient data
                if (
                    consecutive_failures >= max_consecutive_failures
                    and current_data_count >= 3
                ):
                    print_retro_status(
                        "WARNING",
                        f"Forcing completion: {consecutive_failures} failures with {current_data_count} items",
                    )
                    if verbose:
                        print(
                            f"[FORCE] Forcing completion due to {consecutive_failures} consecutive failures with {current_data_count} data items"
                        )
                    store.state.data["achieved"] = True
            else:
                print_retro_status("SUCCESS", "Goal achieved!")
        else:
            print_retro_status("ERROR", f"Unknown action: {decision.action}")
            if verbose:
                print(f"[WARNING] Unknown action: {decision.action}")
            # If unknown action, evaluate to potentially break the loop
            store.dispatch(
                lambda state: goal_evaluation_action(
                    state,
                    model,
                    api_key,
                    tools=store.tools,
                    conversation_history=store.conversation_history,
                    verbose=verbose,
                    store=store,
                ),
                verbose=verbose,
            )

    if store.state.data.get("achieved", False):
        print_retro_banner("MISSION COMPLETE", "★", color=Colors.BRIGHT_GREEN)
        print_retro_status("SUCCESS", "Goal achieved successfully!")
        if verbose:
            print("[SUCCESS] Goal achieved!")
        if output_format:
            print_retro_status("FORMAT", "Formatting final result...")
            if verbose:
                print("[INFO] Formatting final output...")
            try:
                store.dispatch(
                    lambda state: format_output_action(
                        state, model, api_key, output_format, verbose=verbose
                    ),
                    verbose=verbose,
                )

                # Add conversation history to final result
                final_result = store.state.data.get("final_output")
                if final_result:
                    print_retro_status("SUCCESS", "Result formatted successfully!")
                    # Create result with chat history
                    final_result_with_chat = {
                        "result": final_result,
                        "conversation_history": store.conversation_history,
                        "chat_summary": format_conversation_as_chat(
                            store.conversation_history
                        ),
                        "status": "completed_with_formatting",
                    }
                    return final_result_with_chat
                return store.state.data.get("final_output")

            except Exception as e:
                print_retro_status("ERROR", f"Formatting failed: {str(e)}")
                if verbose:
                    print(f"[ERROR] Failed to format final output: {e}")
                    print("[INFO] Returning raw collected data instead")

                # Return collected data even without formatting
                return {
                    "result": None,
                    "raw_data": store.state.data,
                    "conversation_history": store.conversation_history,
                    "chat_summary": format_conversation_as_chat(
                        store.conversation_history
                    ),
                    "status": "completed_without_formatting",
                    "error": f"Formatting failed: {str(e)}",
                }
    else:
        # Determine stop reason
        if consecutive_failures >= max_consecutive_failures:
            error_msg = (
                f"Stopped due to {consecutive_failures} consecutive evaluator failures"
            )
            print_retro_banner("MISSION INTERRUPTED", "!", color=Colors.BRIGHT_RED)
            print_retro_status(
                "ERROR", f"Stopped by {consecutive_failures} consecutive failures"
            )
            if verbose:
                print(f"[WARNING] {error_msg}")
        elif iteration >= max_iterations:
            if crash_if_over_iterations:
                error_msg = "Max iterations reached"
                print_retro_banner("TIME EXPIRED", "!", color=Colors.BRIGHT_RED)
                print_retro_status(
                    "ERROR", f"Limit of {max_iterations} iterations reached - crashing as requested"
                )
                if verbose:
                    print(f"[ERROR] {error_msg}")
                raise RuntimeError(f"Agent exceeded max_iterations ({max_iterations}) and crash_if_over_iterations=True")
            else:
                # Fallback to summarizer on final step
                print_retro_banner("TIME EXPIRED - SUMMARIZING", "!", color=Colors.BRIGHT_YELLOW)
                print_retro_status(
                    "FALLBACK", f"Max iterations ({max_iterations}) reached - calling summarizer to preserve work"
                )
                if verbose:
                    print(f"[FALLBACK] Max iterations reached, running summarizer to preserve work")
                
                # Call summarizer to preserve work done so far
                try:
                    store.dispatch(
                        lambda state: summarize_action(
                            state,
                            model,
                            api_key,
                            tools=store.tools,
                            conversation_history=store.conversation_history,
                            verbose=verbose,
                        ),
                        verbose=verbose,
                    )
                    summary_result = store.state.data.get("summary")
                    if summary_result:
                        print_retro_status("SUCCESS", "Summary generated successfully despite iteration limit")
                        return {
                            "result": summary_result,
                            "conversation_history": store.conversation_history,
                            "chat_summary": format_conversation_as_chat(
                                store.conversation_history
                            ),
                            "status": "completed_with_summary_fallback",
                            "iterations_used": iteration,
                            "max_iterations": max_iterations,
                        }
                except Exception as e:
                    print_retro_status("ERROR", f"Summarizer fallback failed: {str(e)}")
                    if verbose:
                        print(f"[ERROR] Summarizer fallback failed: {e}")
                
                error_msg = "Max iterations reached"
        else:
            error_msg = "Unknown termination reason"
            print_retro_banner("UNEXPECTED STOP", "!", color=Colors.BRIGHT_RED)
            print_retro_status("ERROR", "Unknown stop reason")
            if verbose:
                print(f"[WARNING] {error_msg}")

        # Return history even if not completed
        return {
            "result": None,
            "conversation_history": store.conversation_history,
            "chat_summary": format_conversation_as_chat(store.conversation_history),
            "error": error_msg,
            "final_state": store.state.data,
        }

    return None


# === Example Usage ===
if __name__ == "__main__":
    import time

    # Define a fake tool to fetch weather data with a delay
    def fetch_weather_tool(
        state: Dict[str, Any], args: Dict[str, Any]
    ) -> Optional[Tuple[str, BaseModel]]:
        location = args.get("location", "default")
        print(f"[INFO] Fetching weather for {location}...")
        time.sleep(3)
        # Simulated weather data
        weather_data = {
            "location": location,
            "temperature": "25°C",
            "condition": "Sunny",
        }
        results = state.get("results", []) + [weather_data]
        print(f"[INFO] Weather data fetched for {location}.")
        return ("results", results)

    # Create a dictionary of tools to register
    agent_tools = {"fetch_weather": fetch_weather_tool}

    # Define the desired output format
    class WeatherReport(BaseModel):
        location: str = Field(..., description="The location of the weather report.")
        temperature: str = Field(..., description="The temperature in Celsius.")
        condition: str = Field(..., description="The weather condition.")
        summary: str = Field(..., description="A summary of the weather report.")

    # Create the agent and pass the tools and output format
    agent_goal = "Create a weather report for London."
    final_state = run_agent(
        goal=agent_goal,
        model="ollama/gemma3",
        tools=agent_tools,
        output_format=WeatherReport,
    )
    print("\nFinal State:", final_state)
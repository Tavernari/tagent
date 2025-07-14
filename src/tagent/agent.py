# TAgent main module - orchestrates the agent execution loop.
# Integration with LiteLLM for real LLM calls, leveraging JSON Mode.
# Requirements: pip install pydantic litellm

from typing import Dict, Any, Optional, Callable, Type, Union, Tuple
from pydantic import BaseModel, Field
import json
import time

from .version import __version__
from .store import Store
from .llm_client import query_llm, generate_step_title, generate_step_summary
from .actions import (
    plan_action,
    summarize_action,
    enhanced_goal_evaluation_action,
    format_output_action,
    format_fallback_output_action,
)
from .ui import (
    print_retro_banner,
    print_retro_status,
    print_retro_step,
    print_feedback_dimmed,
    start_thinking,
    stop_thinking,
    Colors,
)
from .utils import detect_action_loop, format_conversation_as_chat
from .state_machine import AgentStateMachine, ActionType, AgentState
from .model_config import AgentModelConfig, AgentStep, ModelConfig, create_config_from_string


class AgentModelSelector:
    """
    Centralized model selector that handles all model selection logic for agent steps.
    
    This class provides a testable abstraction layer for model selection, ensuring
    the correct model is chosen for each agent step based on configuration hierarchy.
    """
    
    def __init__(self, config: AgentModelConfig):
        """
        Initialize the model selector with a configuration object.
        
        Args:
            config: AgentModelConfig object containing model settings
        """
        self.config = config
    
    def get_model_for_step(self, step: AgentStep) -> str:
        """
        Get the appropriate model for a specific agent step.
        
        Args:
            step: The agent step requiring a model
            
        Returns:
            Model name string to use for the specified step
        """
        return ModelConfig.get_model_for_step(
            step=step,
            fallback_model=self.config.tagent_model,
            config=self.config
        )
    
    def get_api_key(self) -> Optional[str]:
        """Get the API key from configuration."""
        return self.config.api_key
    
    def get_planner_model(self) -> str:
        """Get model for planning actions."""
        return self.get_model_for_step(AgentStep.PLANNER)
    
    def get_executor_model(self) -> str:
        """Get model for execution actions."""
        return self.get_model_for_step(AgentStep.EXECUTOR)
    
    def get_summarizer_model(self) -> str:
        """Get model for summarization actions."""
        return self.get_model_for_step(AgentStep.SUMMARIZER)
    
    def get_evaluator_model(self) -> str:
        """Get model for evaluation actions."""
        return self.get_model_for_step(AgentStep.EVALUATOR)
    
    def get_finalizer_model(self) -> str:
        """Get model for final formatting actions."""
        return self.get_model_for_step(AgentStep.FINALIZER)
    
    def get_global_model(self) -> str:
        """Get the global fallback model."""
        return self.config.tagent_model
    
    def __str__(self) -> str:
        """String representation showing the effective models for each step."""
        models = {}
        for step in AgentStep:
            models[step.value] = self.get_model_for_step(step)
        return f"AgentModelSelector(models={models})"


def _generate_and_show_step_summary(
    action: str,
    reasoning: str,
    result_description: str,
    model_selector: AgentModelSelector,
    verbose: bool = False,
) -> None:
    """
    Generate and display a step summary for better user understanding.
    """
    try:
        summary = generate_step_summary(
            action=action,
            reasoning=reasoning,
            result=result_description,
            model=model_selector.get_global_model(),
            api_key=model_selector.get_api_key(),
            verbose=verbose,
        )
        print_feedback_dimmed("STEP_SUMMARY", summary)
    except Exception as e:
        if verbose:
            print(f"[DEBUG] Step summary generation failed: {e}")


def _execute_llm_fallback(
    state: Dict[str, Any],
    tool_name: str,
    tool_args: Dict[str, Any],
    model_selector: AgentModelSelector,
    verbose: bool = False,
) -> Optional[Tuple[str, Any]]:
    """
    Execute a fallback LLM query when a requested tool is not found.
    
    This allows the agent to handle requests that don't require external tools
    by using the LLM's knowledge directly.
    
    Args:
        state: Current agent state
        tool_name: Name of the tool that was requested but not found
        tool_args: Arguments that were intended for the tool
        model_selector: Model selector for LLM access
        verbose: Whether to show debug output
        
    Returns:
        Tuple of (key, value) for state update, or None if failed
    """
    try:
        # Construct a prompt that explains what was requested
        args_text = ", ".join([f"{k}='{v}'" for k, v in tool_args.items()]) if tool_args else "no arguments"
        
        fallback_prompt = (
            f"You are acting as a replacement for a tool called '{tool_name}' that is not available. "
            f"The tool was called with {args_text}. "
            f"Current goal: {state.get('goal', 'No specific goal set')}\n"
            f"Available data: {json.dumps({k: v for k, v in state.items() if k != 'goal'}, indent=2)}\n"
            f"Tool arguments: {json.dumps(tool_args, indent=2) if tool_args else 'None'}\n\n"
            f"Please provide the specific output that the '{tool_name}' tool would have generated. "
            f"Focus on providing direct, actionable content rather than describing what you would do. "
            f"Your response should be the actual result, not a meta-description of the task."
        )
        
        if verbose:
            print(f"[LLM_FALLBACK] Querying LLM for tool '{tool_name}' with prompt: {fallback_prompt[:100]}...")
        
        # For fallback, use a simple text query instead of structured output
        try:
            import litellm
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Provide direct, actionable answers without meta-commentary."},
                {"role": "user", "content": fallback_prompt}
            ]
            
            response = litellm.completion(
                model=model_selector.get_executor_model(),
                messages=messages,
                api_key=model_selector.get_api_key(),
                temperature=0.7
            )
            
            llm_result = response.choices[0].message.content.strip()
            
        except Exception as e:
            if verbose:
                print(f"[LLM_FALLBACK] Direct LLM call failed, trying structured: {e}")
            
            # Fallback to structured response if direct call fails
            response = query_llm(
                fallback_prompt,
                model_selector.get_executor_model(),
                model_selector.get_api_key(),
                tools={},  # No tools for fallback
                conversation_history=[],  # Clean context for focused response
                verbose=verbose,
            )
            
            # Extract the response content - for fallback, we want the reasoning/content, not action
            if hasattr(response, 'reasoning') and response.reasoning:
                llm_result = response.reasoning
            elif hasattr(response, 'action') and response.action:
                llm_result = response.action
            else:
                llm_result = str(response)
        
        # Store the result with a descriptive key
        result_key = f"{tool_name}_llm_response"
        return (result_key, {
            "tool_name": tool_name,
            "requested_args": tool_args,
            "llm_response": llm_result,
            "source": "llm_fallback",
            "timestamp": time.time()
        })
        
    except Exception as e:
        if verbose:
            print(f"[LLM_FALLBACK] Error during fallback execution: {e}")
        return (f"{tool_name}_fallback_error", {
            "tool_name": tool_name,
            "error": str(e),
            "source": "llm_fallback_error"
        })


# === Main Agent Loop ===
def run_agent(
    goal: str,
    config: Optional['TAgentConfig'] = None,
    # Legacy parameters for backward compatibility
    model: Union[str, AgentModelConfig] = "gpt-3.5-turbo",
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
        config: TAgentConfig object containing all configuration options.
                If None, will use legacy parameters and environment variables.
        
        # Legacy parameters (for backward compatibility):
        model: Either a model string (e.g., "gpt-4") for backward compatibility,
            or an AgentModelConfig object for step-specific model configuration.
        api_key: The API key for the LLM service (deprecated - use AgentModelConfig).
        max_iterations: The maximum number of iterations.
        tools: A dictionary of custom tools to register with the agent.
        output_format: The Pydantic model for the final output.
        verbose: If True, shows all debug logs. If False, shows only essential logs.
        crash_if_over_iterations: If True, raises exception when max_iterations
            reached. If False (default), returns results with summarizer fallback.

    Returns:
        An instance of the `output_format` model, or None if no output is generated.
    """
    # Handle configuration: use TAgentConfig if provided, otherwise use legacy parameters
    if config is None:
        # Import here to avoid circular imports
        from .config import TAgentConfig
        
        # Create config from legacy parameters and environment
        config = TAgentConfig(
            model=model,
            api_key=api_key,
            max_iterations=max_iterations,
            tools=tools,
            output_format=output_format,
            verbose=verbose,
            crash_if_over_iterations=crash_if_over_iterations,
        )
        
        # Override with environment variables
        config = config.merge(TAgentConfig.from_env())
    else:
        # Override config with any explicitly provided legacy parameters
        override_dict = {}
        if model != "gpt-3.5-turbo":  # Only override if not default
            override_dict["model"] = model
        if api_key is not None:
            override_dict["api_key"] = api_key
        if max_iterations != 20:  # Only override if not default
            override_dict["max_iterations"] = max_iterations
        if tools is not None:
            override_dict["tools"] = tools
        if output_format is not None:
            override_dict["output_format"] = output_format
        if verbose:  # Only override if True
            override_dict["verbose"] = verbose
        if crash_if_over_iterations:  # Only override if True
            override_dict["crash_if_over_iterations"] = crash_if_over_iterations
        
        if override_dict:
            from .config import TAgentConfig
            config = config.merge(TAgentConfig.from_dict(override_dict))
    
    # Extract values from config
    model_config = config.get_model_config()
    max_iterations = config.max_iterations
    tools = config.tools
    output_format = config.output_format
    verbose = config.verbose
    crash_if_over_iterations = config.crash_if_over_iterations
    
    # Set UI style
    from .ui import set_ui_style
    set_ui_style(config.ui_style)
    
    # Create the centralized model selector
    model_selector = AgentModelSelector(model_config)
    
    # 90s Style Agent Initialization
    from .ui import print_retro_banner, print_retro_status, MessageType
    print_retro_banner(
        f"T-AGENT v{__version__} STARTING", "▓", 60, MessageType.PRIMARY
    )
    print_retro_status("INIT", f"Goal: {goal[:40]}...")
    print_retro_status(
        "CONFIG", f"Model: {model_selector.get_global_model()} | Max Iterations: {max_iterations}"
    )

    store = Store({"goal": goal, "results": [], "used_tools": []})
    
    # Initialize state machine to control valid action transitions
    state_machine = AgentStateMachine()

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
    else:
        # No tools available - check if this is a simple question that can be answered directly
        simple_question_indicators = ["what is", "who is", "where is", "when is", "how is", "which is", "why is"]
        goal_lower = goal.lower()
        
        if any(indicator in goal_lower for indicator in simple_question_indicators):
            print_retro_status("DIRECT_ANSWER", "No tools available - attempting direct LLM response...")
            try:
                import litellm
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Provide direct, accurate answers to questions."},
                    {"role": "user", "content": goal}
                ]
                
                response = litellm.completion(
                    model=model_selector.get_global_model(),
                    messages=messages,
                    api_key=model_selector.get_api_key(),
                    temperature=0.3
                )
                
                direct_answer = response.choices[0].message.content.strip()
                store.state.data["direct_answer"] = direct_answer
                store.state.data["achieved"] = True
                print_retro_status("SUCCESS", "Direct answer provided - goal achieved")
                
            except Exception as e:
                if verbose:
                    print(f"[DEBUG] Direct answer failed: {e}")
                print_retro_status("WARNING", "Direct answer failed - proceeding with normal flow")

    print_retro_banner("STARTING MAIN LOOP", "~", 60, MessageType.SUCCESS)
    iteration = 0
    while (
        state_machine.current_state not in [AgentState.COMPLETED, AgentState.FAILED]
        and iteration < max_iterations
        and consecutive_failures < max_consecutive_failures
    ):
        iteration += 1

        # Add step counting warnings
        remaining_steps = max_iterations - iteration
        if remaining_steps <= 3:
            print_retro_status(
                "WARNING",
                f"Only {remaining_steps} steps remaining before reaching max iterations ({max_iterations})",
            )

        if verbose:
            print(
                f"[LOOP] Iteration {iteration}/{max_iterations}. "
                f"Current state: {store.state.data}"
            )
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
                    f"[PROGRESS] Data items increased from {last_data_count} to "
                    f"{current_data_count} - resetting failure counter"
                )
            consecutive_failures = 0
            last_data_count = current_data_count

        progress_summary = f"Progress: {current_data_count} data items collected"

        used_tools = store.state.data.get("used_tools", [])
        unused_tools = [t for t in store.tools.keys() if t not in used_tools]

        # Check if the last action was 'evaluate' failure
        last_action_was_failed_evaluate = (
            recent_actions
            and recent_actions[-1] == "evaluate"
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
                strategy_hint = (
                    f"Goal evaluation indicates more data is needed. Available tools that haven't been used: {unused_tools}. "
                    "Additional data gathering may help achieve the goal."
                )
            elif last_action == "evaluate" and not unused_tools:
                strategy_hint = (
                    "Goal evaluation shows the current approach needs adjustment. "
                    "Consider planning a new strategy or executing tools with different parameters. "
                    "Gathering more information may help before the next evaluation."
                )
            elif unused_tools:
                strategy_hint = (
                    f"Current pattern may benefit from a different approach. Unused tools available: {unused_tools}. "
                    "Consider planning a new strategy to make progress."
                )
            else:
                strategy_hint = (
                    "Current approach shows repetition. Planning a new strategy or trying different parameters "
                    "may help achieve the goal more effectively."
                )

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
                step_warning = f"⚠️ Only {remaining_steps} step remaining. Focus on goal completion to maximize progress. "
            elif remaining_steps <= 2:
                step_warning = f"⚠️ {remaining_steps} steps remaining. Prioritize actions that advance toward the goal. "
            else:
                step_warning = f"⚠️ {remaining_steps} steps left. Be efficient. "

        # Add instruction to avoid evaluate after failure
        evaluate_guidance = ""
        if last_action_was_failed_evaluate:
            evaluate_guidance = (
                "The last evaluation indicated the goal hasn't been achieved yet. "
                "Consider planning a new strategy based on the feedback, or executing tools to gather more data. "
                "Review missing items and suggestions from the evaluator to guide next steps."
            )

        # Get allowed actions from state machine
        allowed_actions = state_machine.get_allowed_actions(store.state.data)
        allowed_action_names = [action.value for action in allowed_actions]
        
        # Add hint for finalizing
        completion_guidance = ""
        if store.state.data.get("achieved", False) and "finalize" in allowed_action_names:
            completion_guidance = "The goal has been achieved successfully. The 'finalize' action is available to complete the process."

        # Check if only one action is allowed - if so, follow the single path automatically
        if len(allowed_actions) == 1:
            forced_action = list(allowed_actions)[0]
            
            # Special handling when no tools are available and action is EXECUTE
            if forced_action == ActionType.EXECUTE and not store.tools:
                # Check if we already have a direct response to avoid repeating
                if store.state.data.get("llm_direct_response") and store.state.data.get("achieved"):
                    # Goal already achieved with direct response - move to finalize
                    forced_action = ActionType.FINALIZE
                    params = {}
                    print_retro_status("FINALIZE_AUTO", "Goal already achieved - proceeding to finalize")
                else:
                    # Instead of executing nothing, use LLM fallback directly for simple goals
                    simple_question_indicators = ["translate", "what is", "who is", "where is", "when is", "how is", "which is", "why is"]
                    goal_lower = goal.lower()
                    
                    if any(indicator in goal_lower for indicator in simple_question_indicators):
                        print_retro_status("DIRECT_LLM", "No tools available - providing direct LLM response...")
                        try:
                            import litellm
                            messages = [
                                {"role": "system", "content": "You are a helpful assistant. Provide direct, accurate responses."},
                                {"role": "user", "content": goal}
                            ]
                            
                            response = litellm.completion(
                                model=model_selector.get_global_model(),
                                messages=messages,
                                api_key=model_selector.get_api_key(),
                                temperature=0.3
                            )
                            
                            direct_response = response.choices[0].message.content.strip()
                            store.state.data["llm_direct_response"] = direct_response
                            store.state.data["achieved"] = True
                            
                            # Add to conversation history
                            observation = f"Direct LLM response: {direct_response}"
                            store.add_to_conversation("user", observation)
                            
                            print_retro_status("SUCCESS", "Direct response provided - goal achieved")
                            
                            # Force to evaluation to check completion
                            forced_action = ActionType.EVALUATE
                            params = {}
                            
                        except Exception as e:
                            if verbose:
                                print(f"[DEBUG] Direct LLM failed: {e}")
                            print_retro_status("WARNING", "Direct LLM failed - proceeding with summarization")
                            forced_action = ActionType.SUMMARIZE
                            params = {}
                    else:
                        # Not a simple question - force summarization to wrap up
                        forced_action = ActionType.SUMMARIZE
                        params = {}
            else:
                # Normal execution path
                params = {}
                if forced_action == ActionType.EXECUTE:
                    # For execute, we need to pick a tool
                    if unused_tools:
                        params = {"tool": unused_tools[0], "args": {}}
                    elif store.tools:
                        # If all tools used, pick the first one
                        params = {"tool": list(store.tools.keys())[0], "args": {}}
            
            print_retro_status("STATE_AUTO", f"Single path available: {forced_action.value} - following automatically")
                    
            decision = type('MockDecision', (), {
                'action': forced_action.value,
                'params': params,
                'reasoning': f"Following single available path: {forced_action.value}"
            })()
            action_type = forced_action
            skip_llm_query = True
        else:
            # Let AI decide when multiple paths are available
            skip_llm_query = False

            is_summarize_available = ActionType.SUMMARIZE in allowed_actions
            is_evaluate_available = ActionType.EVALUATE in allowed_actions

            prompt = (
                f"Goal: {goal}\n"
                f"Current state: {store.state.data}\n"
                f"{progress_summary}\n"
                f"Step {iteration}/{max_iterations}. {step_warning}\n"
                f"Used tools: {used_tools}\n"
                f"Unused tools: {unused_tools}\n"
                f"{evaluation_feedback}"
                f"{strategy_hint}"
                f"{evaluate_guidance}"
                f"{completion_guidance}"
                "When executing actions, unused tools may provide different types of valuable data. "
                "Evaluation works best after gathering new information or making progress. "
                f"Available actions for this context: {allowed_action_names}. "
                "Consider which action would be most effective for achieving the goal."
            )

            if is_summarize_available:
                prompt += f"\nMandatory: If you want summarize the conversation and, use the {ActionType.SUMMARIZE.value} action."
            if is_evaluate_available:
                prompt += f"\nMandatory: If you want evaluate the conversation, use the {ActionType.EVALUATE.value} action."

            # Add current prompt to history
            store.add_to_conversation("user", prompt)

            print_retro_status("THINKING", "Consulting AI for next action...")
            start_thinking("Thinking")
            try:
                decision = query_llm(
                    prompt,
                    model_selector.get_global_model(),
                    model_selector.get_api_key(),
                    tools=store.tools,
                    conversation_history=store.conversation_history[:-1],
                    verbose=verbose,
                )  # Exclude last message to avoid duplication
            finally:
                stop_thinking()

        # Generate concise step title using LLM
        step_title = generate_step_title(
            decision.action, decision.reasoning, model_selector.get_global_model(), model_selector.get_api_key(), verbose
        )
        print_retro_step(iteration, decision.action, step_title)
        if verbose:
            print(f"[DECISION] LLM decided: {decision}")

        # Validate action with state machine BEFORE tracking (only if not auto-selected)
        if not skip_llm_query:
            try:
                action_type = ActionType(decision.action)
            except ValueError:
                if verbose:
                    print(f"[WARNING] Unknown action type: {decision.action}, defaulting to plan")
                action_type = ActionType.PLAN
                decision.action = "plan"

            # Check if action is allowed by state machine
            if not state_machine.is_action_allowed(action_type, store.state.data):
                forced_action = state_machine.get_forced_action(action_type, store.state.data)
                if verbose:
                    print(f"[STATE_MACHINE] Action {decision.action} not allowed, forcing {forced_action.value}")
                print_retro_status("STATE_CTRL", f"Forced {forced_action.value} (was {decision.action})")
                decision.action = forced_action.value
                decision.reasoning = f"State machine forced {forced_action.value} to prevent invalid transition"
                action_type = forced_action

        # Track recent actions to detect loops (AFTER validation)
        recent_actions.append(decision.action)
        if len(recent_actions) > max_recent_actions:
            recent_actions.pop(0)  # Keep only the latest actions

        # Redirect action if 'evaluate' after previous failure to prevent loops
        if decision.action == "evaluate" and last_action_was_failed_evaluate:
            print_retro_status(
                "WARNING", "Evaluation loop detected - redirecting to planning"
            )
            decision.action = "plan"
            decision.reasoning = "Redirected to planning to address evaluation feedback"
            # Add to history as observation
            store.add_to_conversation(
                "user",
                "Observation: Evaluation loop detected. Redirecting to planning to address evaluator feedback.",
            )

        # Add assistant response to history
        store.add_assistant_response(decision)

        # Dispatch based on LLM decision
        if decision.action == "plan":
            print_retro_status("PLAN", "Generating strategic plan...")
            store.dispatch(
                lambda state: plan_action(
                    state,
                    model_selector.get_planner_model(),
                    model_selector.get_api_key(),
                    tools=store.tools,
                    conversation_history=store.conversation_history,
                    verbose=verbose,
                    config=model_config,
                ),
                verbose=verbose,
            )
            # Generate step summary
            plan_result = f"Strategic plan created with steps and focus areas"
            _generate_and_show_step_summary(
                "plan", decision.reasoning, plan_result, model_selector, verbose
            )
            # Update state machine AFTER successful execution
            state_machine.transition(action_type)
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
                        try:
                            tool_output = json.dumps({key: value}, indent=2)
                        except Exception as e:
                            tool_output = str(value)
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
                # Generate step summary
                tool_result = f"Tool {tool_name} executed - data collected and stored"
                _generate_and_show_step_summary(
                    "execute", decision.reasoning, tool_result, model_selector, verbose
                )
                # Update state machine AFTER successful execution
                state_machine.transition(action_type)
            else:
                print_retro_status("LLM_FALLBACK", f"Tool '{tool_name}' not found - using LLM fallback")
                if verbose:
                    print(
                        f"[LLM_FALLBACK] Tool not found: {tool_name}. Available tools: {list(store.tools.keys())}. Attempting LLM fallback..."
                    )
                
                # Try to fulfill the request using LLM fallback
                fallback_result = _execute_llm_fallback(
                    store.state.data, 
                    tool_name, 
                    tool_args, 
                    model_selector, 
                    verbose
                )
                
                if fallback_result:
                    # Update state with fallback result
                    key, value = fallback_result
                    store.state.data[key] = value
                    
                    # Track that we used a fallback (not a real tool)
                    used_tools = store.state.data.get("used_tools", [])
                    fallback_tool_name = f"{tool_name}_llm_fallback"
                    if fallback_tool_name not in used_tools:
                        used_tools.append(fallback_tool_name)
                        store.state.data["used_tools"] = used_tools
                    
                    # Add result to conversation history
                    try:
                        fallback_output = json.dumps(value, indent=2)
                    except Exception:
                        fallback_output = str(value)
                    
                    observation = f"LLM Fallback for tool '{tool_name}': {fallback_output}"
                    store.add_to_conversation("user", observation)
                    
                    print_retro_status(
                        "SUCCESS",
                        f"LLM fallback for '{tool_name}' completed successfully"
                    )
                    # Generate step summary for fallback
                    fallback_result_desc = f"LLM fallback for tool '{tool_name}' executed - response generated and stored"
                    _generate_and_show_step_summary(
                        "execute", decision.reasoning, fallback_result_desc, model_selector, verbose
                    )
                    # Update state machine AFTER successful fallback execution
                    state_machine.transition(action_type)
                else:
                    # Fallback failed, use original error behavior
                    observation = f"Error: Tool '{tool_name}' not found and LLM fallback failed."
                    store.add_to_conversation("user", observation)
                    print_retro_status("ERROR", f"Tool '{tool_name}' not found and fallback failed")
        elif decision.action == "summarize":
            print_retro_status("SUMMARIZE", "Generating progress summary...")
            store.dispatch(
                lambda state: summarize_action(
                    state,
                    model_selector.get_summarizer_model(),
                    model_selector.get_api_key(),
                    tools=store.tools,
                    conversation_history=store.conversation_history,
                    verbose=verbose,
                    config=model_config,
                ),
                verbose=verbose,
            )
            # Generate step summary
            summary_result = f"Summary generated combining {len(store.conversation_history)} conversation items"
            _generate_and_show_step_summary(
                "summarize", decision.reasoning, summary_result, model_selector, verbose
            )
            # Update state machine AFTER successful execution
            state_machine.transition(action_type)
            
            # After summarize, automatically run evaluate to check if goal was achieved
            if store.state.data.get("summary"):
                print_retro_status("AUTO_EVAL", "Auto-evaluating after summarization...")
                state_machine.transition(ActionType.EVALUATE)  # Update state machine for auto-eval
                store.dispatch(
                    lambda state: enhanced_goal_evaluation_action(
                        state,
                        model_selector.get_evaluator_model(),
                        model_selector.get_api_key(),
                        tools=store.tools,
                        conversation_history=store.conversation_history,
                        verbose=verbose,
                        store=store,
                        config=model_config,
                    ),
                    verbose=verbose,
                )
        elif decision.action == "evaluate":
            print_retro_status("EVALUATE", "Evaluating if goal was achieved...")
            # Store previous state to detect change
            previous_achieved = store.state.data.get("achieved", False)
            store.dispatch(
                lambda state: enhanced_goal_evaluation_action(
                    state,
                    model_selector.get_evaluator_model(),
                    model_selector.get_api_key(),
                    tools=store.tools,
                    conversation_history=store.conversation_history,
                    verbose=verbose,
                    store=store,
                    config=model_config,
                ),
                verbose=verbose,
            )
            # Generate step summary
            eval_achieved = store.state.data.get("achieved", False)
            eval_result = f"Goal evaluation: {'achieved' if eval_achieved else 'not yet achieved'}"
            _generate_and_show_step_summary(
                "evaluate", decision.reasoning, eval_result, model_selector, verbose
            )
            # Update state machine AFTER successful execution
            state_machine.transition(action_type)

            # Check evaluation result and get detailed feedback
            current_achieved = store.state.data.get("achieved", False)
            evaluation_result = store.state.data.get("evaluation_result", {})

            if current_achieved:
                 print_retro_status("SUCCESS", "Goal achieved! Ready to finalize.")
            elif not current_achieved and not previous_achieved:
                consecutive_failures += 1
                evaluation_rejections += 1

                # Extract and show specific feedback from evaluator
                feedback = evaluation_result.get(
                    "feedback", "No specific feedback provided"
                )
                missing_items = evaluation_result.get("missing_items", [])
                suggestions = evaluation_result.get("suggestions", [])
                
                # Auto-trigger PLAN after failed evaluation to use feedback
                print_retro_status("AUTO_PLAN", "Auto-planning after evaluation feedback...")
                store.dispatch(
                    lambda state: plan_action(
                        state,
                        model_selector.get_planner_model(),
                        model_selector.get_api_key(),
                        tools=store.tools,
                        conversation_history=store.conversation_history,
                        verbose=verbose,
                        config=model_config,
                    ),
                    verbose=verbose,
                )
                state_machine.transition(ActionType.PLAN)  # Update state machine AFTER execution

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
                            missing_strings = [
                                str(item) if not isinstance(item, str) else item
                                for item in missing_items
                            ]
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
                            missing_strings = [
                                str(item) if not isinstance(item, str) else item
                                for item in missing_items
                            ]
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
                    # Execute plan_action to address feedback
                    store.dispatch(
                        lambda state: plan_action(
                            state,
                            model_selector.get_planner_model(),
                            model_selector.get_api_key(),
                            tools=store.tools,
                            conversation_history=store.conversation_history,
                            verbose=verbose,
                            config=model_config,
                        ),
                        verbose=verbose,
                    )
                    recent_actions.append("plan")  # Update to break loop

                # After 2 evaluation failures, encourage alternative actions
                if consecutive_failures >= 2:
                    if verbose:
                        print_retro_status(
                            "WARNING",
                            "Too many evaluation failures - strategy change needed",
                        )
                    # Skip the next evaluate decision by manipulating recent actions
                    recent_actions.append(
                        "evaluate"
                    )  # Add extra evaluate to trigger loop detection

                # Consider completion when many failures occur but sufficient data exists
                if (
                    consecutive_failures >= max_consecutive_failures
                    and current_data_count >= 3
                ):
                    print_retro_status(
                        "WARNING",
                        f"Auto-completion triggered: {consecutive_failures} failures with {current_data_count} items",
                    )
                    if verbose:
                        print(
                            f"[AUTO] Auto-completion due to {consecutive_failures} consecutive failures with {current_data_count} data items"
                        )
                    store.state.data["achieved"] = True
        elif decision.action == "finalize":
            print_retro_status("FINALIZE", "Finalizing the result...")
            state_machine.transition(action_type)  # To FINALIZING
            
            if output_format:
                try:
                    store.dispatch(
                        lambda state: format_output_action(
                            state,
                            model_selector.get_finalizer_model(),
                            model_selector.get_api_key(),
                            output_format,
                            verbose=verbose,
                            config=model_config,
                            conversation_history=store.conversation_history,
                        ),
                        verbose=verbose,
                    )
                    state_machine.current_state = AgentState.COMPLETED
                except Exception as e:
                    print_retro_status("ERROR", f"Finalizing with format failed: {str(e)}")
                    if verbose:
                        print(f"[ERROR] Finalizing with format failed: {e}")
                    state_machine.current_state = AgentState.FAILED  # Fail if formatting fails
            else:
                # No output format, so we can consider it completed.
                state_machine.current_state = AgentState.COMPLETED
        else:
            print_retro_status("ERROR", f"Unknown action: {decision.action}")
            if verbose:
                print(f"[WARNING] Unknown action: {decision.action}")
            # If unknown action, evaluate to potentially break the loop
            store.dispatch(
                lambda state: enhanced_goal_evaluation_action(
                    state,
                    model_selector.get_evaluator_model(),
                    model_selector.get_api_key(),
                    tools=store.tools,
                    conversation_history=store.conversation_history,
                    verbose=verbose,
                    store=store,
                    config=model_config,
                ),
                verbose=verbose,
            )

    if state_machine.current_state == AgentState.COMPLETED:
        print_retro_banner("MISSION COMPLETE", "★", 60, MessageType.SUCCESS)
        print_retro_status("SUCCESS", "Goal achieved successfully!")
        if verbose:
            print("[SUCCESS] Goal achieved!")
        
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
                "iterations_used": iteration,
                "max_iterations": max_iterations,
            }
            return final_result_with_chat
        elif output_format:
             # Formatting failed, but we have an output format
            print_retro_status("ERROR", "Formatting failed, returning raw data.")
            return {
                "result": None,
                "raw_data": store.state.data,
                "conversation_history": store.conversation_history,
                "chat_summary": format_conversation_as_chat(
                    store.conversation_history
                ),
                "status": "completed_without_formatting",
                "error": "Formatting failed, but an output format was provided.",
                "iterations_used": iteration,
                "max_iterations": max_iterations,
            }
        else:
            # No output format specified, return raw collected data
            print_retro_status("SUCCESS", "Goal achieved! Returning collected data.")
            return {
                "result": None,
                "raw_data": store.state.data,
                "conversation_history": store.conversation_history,
                "chat_summary": format_conversation_as_chat(
                    store.conversation_history
                ),
                "status": "completed_without_formatting",
                "iterations_used": iteration,
                "max_iterations": max_iterations,
            }
    else:
        # Determine stop reason
        if state_machine.current_state == AgentState.FAILED:
            error_msg = "Agent failed during finalization."
            print_retro_banner("MISSION FAILED", "!", 60, MessageType.ERROR)
            print_retro_status("ERROR", error_msg)
        elif consecutive_failures >= max_consecutive_failures:
            error_msg = (
                f"Stopped due to {consecutive_failures} consecutive evaluator failures"
            )
            print_retro_banner("MISSION INTERRUPTED", "!", 60, MessageType.ERROR)
            print_retro_status(
                "ERROR", f"Stopped by {consecutive_failures} consecutive failures"
            )
            if verbose:
                print(f"[WARNING] {error_msg}")
        elif iteration >= max_iterations:
            if crash_if_over_iterations:
                error_msg = "Max iterations reached"
                print_retro_banner("TIME EXPIRED", "!", 60, MessageType.ERROR)
                print_retro_status(
                    "ERROR",
                    f"Limit of {max_iterations} iterations reached - crashing as requested",
                )
                if verbose:
                    print(f"[ERROR] {error_msg}")
                raise RuntimeError(
                    f"Agent exceeded max_iterations ({max_iterations}) and crash_if_over_iterations=True"
                )
            else:
                # Fallback to summarizer on final step
                print_retro_banner(
                    "TIME EXPIRED - SUMMARIZING", "!", color=Colors.BRIGHT_YELLOW
                )
                print_retro_status(
                    "FALLBACK",
                    f"Max iterations ({max_iterations}) reached - calling summarizer to preserve work",
                )
                if verbose:
                    print(
                        f"[FALLBACK] Max iterations reached, running summarizer to preserve work"
                    )

                # Call summarizer to preserve work done so far
                summary_result = None
                evaluation_result = {}
                formatted_result = None
                
                try:
                    # Update state machine to reflect we're moving to SUMMARIZE
                    print_retro_status("STATE_AUTO", "Max iterations reached - forcing SUMMARIZE action")
                    state_machine.transition(ActionType.SUMMARIZE)
                    
                    store.dispatch(
                        lambda state: summarize_action(
                            state,
                            model_selector.get_summarizer_model(),
                            model_selector.get_api_key(),
                            tools=store.tools,
                            conversation_history=store.conversation_history,
                            verbose=verbose,
                            config=model_config,
                        ),
                        verbose=verbose,
                    )
                    summary_result = store.state.data.get("summary")
                    
                    # Always try to run evaluation, regardless of summary success
                    print_retro_status("AUTO_EVAL", "Auto-evaluating after summarization...")
                    try:
                        state_machine.transition(ActionType.EVALUATE)  # Update state machine for auto-eval
                        store.dispatch(
                            lambda state: enhanced_goal_evaluation_action(
                                state,
                                model_selector.get_evaluator_model(),
                                model_selector.get_api_key(),
                                tools=store.tools,
                                conversation_history=store.conversation_history,
                                verbose=verbose,
                                store=store,
                                config=model_config,
                            ),
                            verbose=verbose,
                        )
                        evaluation_result = store.state.data.get("evaluation", {})
                    except Exception as e:
                        print_retro_status("WARNING", f"Auto-evaluation failed: {str(e)}")
                        if verbose:
                            print(f"[WARNING] Auto-evaluation failed: {e}")

                    # Always try to format output if output_format is provided
                    if output_format:
                        try:
                            print_retro_status("FORMAT_FALLBACK", "Applying output schema to available data...")
                            store.dispatch(
                                lambda state: format_fallback_output_action(
                                    state,
                                    model_selector.get_finalizer_model(),
                                    model_selector.get_api_key(),
                                    output_format,
                                    verbose=verbose,
                                    config=model_config,
                                    conversation_history=store.conversation_history,  # Pass history
                                ),
                                verbose=verbose,
                            )
                            formatted_result = store.state.data.get("final_output")
                            print_retro_status("SUCCESS", "Output structured despite incomplete goal!")
                        except Exception as e:
                            print_retro_status("WARNING", f"Fallback formatting failed: {str(e)}")
                            if verbose:
                                print(f"[WARNING] Fallback formatting failed: {e}")
                        
                except Exception as e:
                    print_retro_status("ERROR", f"Summarizer fallback failed: {str(e)}")
                    if verbose:
                        print(f"[ERROR] Summarizer fallback failed: {e}")

                # Always return available context, even if summarizer/evaluator failed
                print_retro_status("CONTEXT_OUTPUT", "Generating output with available context...")
                return {
                    "result": formatted_result or summary_result or "Max iterations reached - returning available context",
                    "evaluation": evaluation_result,
                    "raw_data": store.state.data,
                    "conversation_history": store.conversation_history,
                    "chat_summary": format_conversation_as_chat(
                        store.conversation_history
                    ),
                    "status": "completed_with_summary_fallback",
                    "iterations_used": iteration,
                    "max_iterations": max_iterations,
                    "formatted_output": formatted_result is not None,
                    "summary_generated": summary_result is not None,
                    "evaluation_completed": bool(evaluation_result),
                }
        else:
            error_msg = "Unknown termination reason"
            print_retro_banner("UNEXPECTED STOP", "!", 60, MessageType.ERROR)
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
            "iterations_used": iteration,
            "max_iterations": max_iterations,
        }


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

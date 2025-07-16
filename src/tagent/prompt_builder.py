"""
Prompt Builder for TAgent

This module constructs prompts for the LLM by combining instructions from the 
InstructionsRAG with dynamic context from the agent's state.
"""

from typing import Dict, Any, Optional

from .instructions_rag import InstructionsRAG
from .memory_manager import EnhancedContextManager
from .state_machine import TaskBasedStateMachine


class PromptBuilder:
    """Builds prompts for different agent actions."""

    def __init__(self, instructions_rag: InstructionsRAG):
        self.instructions_rag = instructions_rag

    def build_planning_prompt(self, state_machine: TaskBasedStateMachine, context_manager: EnhancedContextManager) -> str:
        """Builds the prompt for the task-based planning action."""
        planning_context = state_machine.get_planning_context()
        rag_context_str = context_manager.get_context_for_current_state(state_machine.state)
        
        prompt_parts = [
            f"GOAL: {planning_context['goal']}",
            f"PLANNING CYCLE: {planning_context['planning_cycle']}",
        ]

        if planning_context.get('failed_tasks_context'):
            prompt_parts.append("PREVIOUS FAILURES TO ADDRESS:")
            prompt_parts.append(planning_context['failed_tasks_context'])

        if planning_context.get('collected_data'):
            from .token_utils import truncate_text_by_tokens
            prompt_parts.append("EXISTING DATA:")
            for key, value in planning_context['collected_data'].items():
                value_str = truncate_text_by_tokens(str(value), max_tokens=250)
                prompt_parts.append(f"- {key}: {value_str}")

        if rag_context_str:
            prompt_parts.append(f"\n{rag_context_str}")

        # Get instructions from RAG, which now include tool definitions
        instructions = self.instructions_rag.get_instructions_for_action("plan", planning_context)
        if instructions:
            prompt_parts.append("\nINSTRUCTIONS:")
            prompt_parts.extend(instructions)

        return "\n".join(prompt_parts)

    def build_execution_prompt(self, state_machine: TaskBasedStateMachine, task_description: str, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Builds the prompt for the LLM fallback execution action."""
        exec_context = {
            "goal": state_machine.state.goal,
            "current_data": state_machine.state.collected_data,
            "task_description": task_description,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "llm_fallback": True,  # Trigger for instruction retrieval
        }

        instructions = self.instructions_rag.get_instructions_for_action("execute", exec_context)
        
        prompt_parts = [
            f"TASK: {task_description}",
            f"GOAL: {state_machine.state.goal}",
            f"CURRENT DATA: {state_machine.state.collected_data}",
            f"TASK ARGUMENTS: {tool_args}",
            "\nINSTRUCTIONS:",
            *instructions
        ]
        
        return "\n".join(prompt_parts)

    def build_evaluation_prompt(self, state_machine: TaskBasedStateMachine, context_manager: EnhancedContextManager) -> str:
        """Builds the prompt for the task-based evaluation action."""
        eval_context = state_machine.get_evaluation_context()
        rag_context_str = context_manager.get_context_for_current_state(state_machine.state)

        prompt_parts = [
            f"GOAL: {eval_context['goal']}",
            "TASK EXECUTION SUMMARY:",
            f"- Total tasks: {eval_context['total_tasks']}",
            f"- Completed: {eval_context['completed_tasks']}",
            f"- Failed: {eval_context['failed_tasks']}",
        ]

        if eval_context.get('collected_data'):
            from .token_utils import truncate_text_by_tokens
            prompt_parts.append("COLLECTED DATA:")
            for key, value in eval_context['collected_data'].items():
                value_str = truncate_text_by_tokens(str(value), max_tokens=250)
                prompt_parts.append(f"- {key}: {value_str}")

        if eval_context.get('failed_tasks_details'):
            prompt_parts.append("FAILED TASKS DETAILS:")
            for failed_task in eval_context['failed_tasks_details']:
                prompt_parts.append(f"- {failed_task['description']}: {failed_task['failure_reason']}")

        if rag_context_str:
            prompt_parts.append(f"\n{rag_context_str}")

        instructions = self.instructions_rag.get_instructions_for_action("evaluate", eval_context)
        if instructions:
            prompt_parts.append("\nINSTRUCTIONS:")
            prompt_parts.extend(instructions)

        return "\n".join(prompt_parts)

    def build_finalize_prompt(self, state_machine: TaskBasedStateMachine, context_manager: EnhancedContextManager, output_format: Optional[Any] = None) -> str:
        """Builds the prompt for the task-based finalization action."""
        state_summary = state_machine.get_state_summary()
        memory_summary = context_manager.get_memory_summary_for_finalize()

        prompt_parts = [
            f"GOAL: {state_summary['goal']}",
            "EXECUTION SUMMARY:",
            f"- Planning cycles: {state_summary['planning_cycles']}",
            f"- Total tasks: {state_summary['task_summary']['total_tasks']}",
            f"- Completed tasks: {state_summary['task_summary']['completed']}",
            f"- Failed tasks: {state_summary['task_summary']['failed']}",
        ]

        if state_machine.state.collected_data:
            from .token_utils import truncate_text_by_tokens
            prompt_parts.append("COLLECTED DATA:")
            for key, value in state_machine.state.collected_data.items():
                value_str = truncate_text_by_tokens(str(value), max_tokens=75)
                prompt_parts.append(f"- {key}: {value_str}")

        if memory_summary.get('key_facts'):
            prompt_parts.append("KEY FACTS:")
            for fact in memory_summary['key_facts'][:10]:
                prompt_parts.append(f"- {fact}")

        if memory_summary.get('execution_results'):
            prompt_parts.append("EXECUTION HISTORY:")
            for result in memory_summary['execution_results'][:10]:
                prompt_parts.append(f"- {result['action']}: {result['content']}")

        finalize_context = {"output_format": bool(output_format)}
        if not output_format:
            finalize_context["default_output"] = True

        instructions = self.instructions_rag.get_instructions_for_action("finalize", finalize_context)
        if instructions:
            prompt_parts.append("\nINSTRUCTIONS:")
            prompt_parts.extend(instructions)

        if output_format:
            from pydantic import BaseModel
            if issubclass(output_format, BaseModel):
                schema = output_format.model_json_schema()
                prompt_parts.append(f"\nOUTPUT FORMAT: {schema}")

        return "\n".join(prompt_parts)
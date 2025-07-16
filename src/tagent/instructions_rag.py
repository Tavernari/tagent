"""
Instructions RAG for TAgent

This module provides a specialized RAG implementation for storing and retrieving
the instructional prompts that guide the agent's behavior for different actions.
It now also orchestrates the inclusion of tool definitions.
"""

from typing import Dict, List, Any, Optional
from .semantic_search import RagDocument, TFIDFRagContextManager
from .tool_rag import ToolRAG


class InstructionsRAG(TFIDFRagContextManager):
    """
    A RAG manager for agent instructions, now also responsible for injecting
    tool definitions (including JSON schemas) into the context.
    """

    def __init__(self, tool_rag: Optional[ToolRAG] = None):
        """
        Initialize the RAG with all agent instructions.
        
        Args:
            tool_rag: An optional instance of ToolRAG to get tool definitions from.
        """
        super().__init__(goal="agent_instructions")
        self.tool_rag = tool_rag
        self._initialize_instructions()

    def _initialize_instructions(self):
        """Load all hardcoded instructions into RAG documents."""
        instructions = [
            # Planning Instructions
            RagDocument(
                id="plan_base",
                content="Create a complete and comprehensive list of all tasks required to achieve the goal. Focus on describing *what* needs to be done. Ensure the tasks are logically ordered.",
                doc_type="instruction",
                keywords=["plan", "planning", "base", "strategy"],
                metadata={"action": "plan"}
            ),
            RagDocument(
                id="plan_with_failures",
                content="There were previous failures. Create tasks that specifically address these issues. Analyze the failure reasons and devise a new plan to overcome them.",
                doc_type="instruction",
                keywords=["plan", "planning", "failures", "retry", "address_issues"],
                metadata={"action": "plan", "context_trigger": "failed_tasks_context"}
            ),
            RagDocument(
                id="plan_select_and_use",
                content="If a previous task returned a list of items (e.g., articles, files), the next task MUST process one of those items. You must explicitly select one item from the `EXISTING DATA` and pass its properties (e.g., a URL) as an argument to the next tool. Do not invent values or create a separate 'selection' task.",
                doc_type="instruction",
                keywords=["plan", "planning", "select", "use", "data", "context"],
                metadata={"action": "plan", "context_trigger": "collected_data"}
            ),
            RagDocument(
                id="plan_use_output",
                content="To use the output of a previous task as an argument for a subsequent task, you MUST use the `{{task_ID.output}}` placeholder. For example, if `task_2` loads the content of an article, and `task_3` needs to summarize it, the `text` argument for `task_3` must be `{{task_2.output}}`.",
                doc_type="instruction",
                keywords=["plan", "planning", "output", "placeholder", "argument"],
                metadata={"action": "plan"}
            ),
            RagDocument(
                id="plan_must_select_tool",
                content="For each task, you MUST select the most appropriate tool from the `AVAILABLE TOOLS` list and populate the `tool_name` field. If no specific tool is suitable for the task, you MUST use the built-in `llm_task` tool.",
                doc_type="instruction",
                keywords=["plan", "planning", "tool", "select", "must"],
                metadata={"action": "plan", "context_trigger": "has_tools"}
            ),
            RagDocument(
                id="plan_tool_schema",
                content="When defining task arguments, you MUST adhere to the `Input Schema` provided for each tool. The arguments must be a valid JSON object matching that schema.",
                doc_type="instruction",
                keywords=["plan", "planning", "tool", "schema", "input"],
                metadata={"action": "plan", "context_trigger": "has_tools"}
            ),

            # Execution Instructions
            RagDocument(
                id="execute_llm_fallback",
                content="The requested tool is not available, but you can complete this task using your knowledge directly. Provide the result that would be expected from this task. Focus on providing the actual output needed to accomplish the goal, not a description of what you would do. Be direct and provide the specific result requested.",
                doc_type="instruction",
                keywords=["execute", "llm_fallback", "no_tool", "direct_knowledge"],
                metadata={"action": "execute", "context_trigger": "llm_fallback"}
            ),

            # Evaluation Instructions
            RagDocument(
                id="evaluate_base",
                content="Evaluate whether the *entire* goal has been achieved by comparing the `COLLECTED DATA` and `TASK EXECUTION SUMMARY` against the `GOAL`. Do not assume the goal is complete just because the planned tasks are done. Verify that the data reflects the full completion of the goal. If parts of the goal are not yet addressed, the goal is not achieved.",
                doc_type="instruction",
                keywords=["evaluate", "evaluation", "base", "assessment"],
                metadata={"action": "evaluate"}
            ),
            RagDocument(
                id="evaluate_goal_not_achieved",
                content="If the goal is not achieved, provide specific reasons and suggestions for improvement.",
                doc_type="instruction",
                keywords=["evaluate", "evaluation", "goal_not_achieved", "suggestions", "improvement"],
                metadata={"action": "evaluate"}
            ),

            # Finalize Instructions
            RagDocument(
                id="finalize_base",
                content="Create a comprehensive final output that summarizes the entire task-based execution. Include what was achieved, what challenges were encountered, and the final results. Use all the collected data and memories to create a thorough summary.",
                doc_type="instruction",
                keywords=["finalize", "finalization", "summary", "base", "comprehensive_output"],
                metadata={"action": "finalize"}
            ),
            RagDocument(
                id="finalize_with_schema",
                content="Format your response according to the provided JSON schema.",
                doc_type="instruction",
                keywords=["finalize", "finalization", "schema", "json", "format"],
                metadata={"action": "finalize", "context_trigger": "output_format"}
            ),
             RagDocument(
                id="finalize_default_result",
                content="IMPORTANT: The 'result' field should contain the main answer/output that directly addresses the user's goal. This is what the user is primarily looking for.",
                doc_type="instruction",
                keywords=["finalize", "finalization", "result_field", "main_answer"],
                metadata={"action": "finalize", "context_trigger": "default_output"}
            ),
        ]
        for doc in instructions:
            self.documents[doc.id] = doc
        
        # Fit the vectorizer with the loaded instructions
        self._update_embeddings()

    def get_instructions_for_action(self, action: str, context: Dict[str, Any]) -> List[str]:
        """
        Retrieves relevant instructions for a given action and context, including tool definitions.
        
        Args:
            action: The name of the action (e.g., 'plan', 'execute').
            context: The current context dictionary to help select instructions.
            
        Returns:
            A list of instruction strings, potentially including formatted tool definitions.
        """
        # Build a rich query from the context
        query_parts = [action]
        query_parts.extend(context.keys())
        
        for key, value in context.items():
            if isinstance(value, str):
                query_parts.append(value)
            elif isinstance(value, list):
                query_parts.extend([str(v) for v in value if isinstance(v, (str, int, float))])

        query = " ".join(query_parts)
        
        # Perform semantic search
        search_results = self.semantic_search(query, top_k=5, doc_types=["instruction"])
        
        # Filter results to match the action and context triggers
        relevant_instructions = []
        seen_instructions = set()

        # Inject tool definitions for the 'plan' action
        if action == "plan" and self.tool_rag:
            tool_defs = self.tool_rag.get_all_tools_definitions_for_prompt()
            if tool_defs:
                relevant_instructions.append("AVAILABLE TOOLS:\n" + tool_defs)
                context["has_tools"] = True # For context_trigger

        for doc, score in search_results:
            if doc.metadata.get("action") == action:
                trigger = doc.metadata.get("context_trigger")
                if not trigger or context.get(trigger):
                    if doc.content not in seen_instructions:
                        relevant_instructions.append(doc.content)
                        seen_instructions.add(doc.content)
                    
        # Ensure base instructions are included if not found by search
        base_instruction_map = {
            "plan": "plan_base",
            "evaluate": "evaluate_base",
            "finalize": "finalize_base",
        }
        if action in base_instruction_map:
            base_doc = self.get_document_by_id(base_instruction_map[action])
            if base_doc and base_doc.content not in seen_instructions:
                 relevant_instructions.insert(0, base_doc.content)

        return relevant_instructions
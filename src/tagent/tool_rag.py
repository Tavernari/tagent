"""
Tool RAG for TAgent

This module provides a RAG implementation for dynamically finding and selecting
the appropriate tool for a given task description. It now performs type-aware
introspection to extract Pydantic models and generate JSON schemas for tools.
"""

import inspect
import json
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Callable
from .semantic_search import RagDocument, TFIDFRagContextManager


class ToolRAG(TFIDFRagContextManager):
    """
    A RAG manager specialized for storing and retrieving tools.
    It inspects tool signatures to find Pydantic models and uses their
    JSON schemas to provide structured input formats.
    """

    def __init__(self, tools: Dict[str, Callable]):
        """
        Initialize the ToolRAG with all available tools.
        
        Args:
            tools: A dictionary of tool names to their callable functions.
        """
        super().__init__(goal="agent_tools")
        self._initialize_tools(tools)

    def _initialize_tools(self, tools: Dict[str, Callable]):
        """
        Load all tools into RAG documents, inspecting their signature
        for Pydantic models to generate input and output schemas.
        """
        tool_documents = []
        for tool_name, tool_func in tools.items():
            docstring = tool_func.__doc__
            if not docstring:
                print(f"Warning: Tool '{tool_name}' has no docstring and will be ignored by ToolRAG.")
                continue

            input_schema = None
            output_schema = None
            signature_str = ""
            try:
                # Introspect the function signature
                sig = inspect.signature(tool_func)
                signature_str = str(sig)

                # Find input schema from parameters
                for param in sig.parameters.values():
                    if inspect.isclass(param.annotation) and issubclass(param.annotation, BaseModel) and param.annotation is not BaseModel:
                        input_schema = param.annotation.model_json_schema()
                        break
                
                # Find output schema from return annotation
                if inspect.isclass(sig.return_annotation) and issubclass(sig.return_annotation, BaseModel) and sig.return_annotation is not BaseModel:
                    output_schema = sig.return_annotation.model_json_schema()

            except (ValueError, TypeError):
                print(f"Warning: Could not inspect signature for tool '{tool_name}'.")

            tool_documents.append(
                RagDocument(
                    id=tool_name,
                    content=docstring.strip(),
                    doc_type="tool",
                    keywords=[],
                    metadata={
                        "tool_name": tool_name,
                        "input_schema": input_schema,
                        "output_schema": output_schema,
                        "signature": signature_str
                    }
                )
            )
        
        for doc in tool_documents:
            self.documents[doc.id] = doc
        
        if tool_documents:
            self._update_embeddings()

        self._add_system_tools()

    def _add_system_tools(self):
        """Adds built-in system tools like the LLM fallback tool."""
        llm_task_doc = RagDocument(
            id="llm_task",
            content="A general-purpose tool for tasks requiring reasoning, knowledge, or when no other specific tool is appropriate. Use this for summarization, creative writing, general questions, or complex analysis.",
            doc_type="tool",
            keywords=["llm", "reasoning", "knowledge", "fallback", "general"],
            metadata={
                "tool_name": "llm_task",
                "input_schema": {
                    "title": "LlmTaskInput",
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "title": "Prompt",
                            "description": "A clear and specific prompt describing what the LLM should do.",
                            "type": "string"
                        }
                    },
                    "required": ["prompt"]
                },
                "output_schema": {
                    "title": "LlmTaskOutput",
                    "type": "object",
                    "properties": {
                        "response": {
                            "title": "Response",
                            "description": "The text-based response from the LLM.",
                            "type": "string"
                        }
                    }
                },
                "signature": "(prompt: str) -> str"
            }
        )
        self.documents[llm_task_doc.id] = llm_task_doc
        # Re-fit the vectorizer with the new tool
        self._update_embeddings()

    def find_tool_for_task(self, task_description: str, top_k: int = 1) -> Optional[str]:
        """
        Finds the best tool for a given task description using semantic search.
        
        Args:
            task_description: The natural language description of the task.
            top_k: The number of top results to consider. Defaults to 1.
            
        Returns:
            The name of the best-matching tool, or None if no suitable tool is found.
        """
        if not self.documents:
            return None

        search_results = self.semantic_search(task_description, top_k=top_k, doc_types=["tool"])
        
        if not search_results:
            return None
            
        # The top result is the most likely tool
        best_tool_doc, _ = search_results[0]
        return best_tool_doc.metadata.get("tool_name")

    def get_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full definition for a specific tool, including schema.
        
        Args:
            tool_name: The name of the tool.
            
        Returns:
            A dictionary with the tool's definition, or None if not found.
        """
        tool_doc = self.documents.get(tool_name)
        if tool_doc:
            return {
                "name": tool_name,
                "description": tool_doc.content,
                "signature": tool_doc.metadata.get("signature"),
                "input_schema": tool_doc.metadata.get("input_schema"),
                "output_schema": tool_doc.metadata.get("output_schema"),
            }
        return None

    def get_all_tools_definitions_for_prompt(self) -> str:
        """
        Returns a single string containing the definitions of all tools,
        formatted for inclusion in a prompt.
        """
        definitions = []
        for doc_id in self.documents:
            if self.documents[doc_id].doc_type == "tool":
                definition = self.get_tool_definition(doc_id)
                if definition:
                    formatted_def = (
                        f"Tool: {definition['name']}{definition['signature']}\n"
                        f"Description: {definition['description']}\n"
                    )
                    if definition['input_schema']:
                        formatted_def += f"Input Schema: {json.dumps(definition['input_schema'], indent=2)}\n"
                    if definition['output_schema']:
                        formatted_def += f"Output Schema: {json.dumps(definition['output_schema'], indent=2)}\n"
                    definitions.append(formatted_def)
        
        return "\n---\n".join(definitions)

    def get_tool_docstring(self, tool_name: str) -> Optional[str]:
        """DEPRECATED: Use get_tool_definition instead."""
        print("Warning: get_tool_docstring is deprecated. Use get_tool_definition instead.")
        definition = self.get_tool_definition(tool_name)
        return definition['description'] if definition else None

    def get_all_tools_docstrings(self) -> str:
        """DEPRECATED: Use get_all_tools_definitions_for_prompt instead."""
        print("Warning: get_all_tools_docstrings is deprecated. Use get_all_tools_definitions_for_prompt instead.")
        return self.get_all_tools_definitions_for_prompt()
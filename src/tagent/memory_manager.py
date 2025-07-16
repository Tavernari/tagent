"""
Context Manager for TAgent - Imports real RAG implementation with TF-IDF vectorization.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .models import MemoryItem, EnhancedAgentState
from .semantic_search import TFIDFRagContextManager, RagDocument


class SimpleRagContextManager:
    """
    Simple RAG Context Manager for TAgent.
    
    Uses in-memory storage with basic keyword and semantic matching.
    This is a minimal implementation that can be extended with vector databases later.
    """
    
    def __init__(self, goal: str):
        self.goal = goal
        self.documents: Dict[str, RagDocument] = {}
        self.doc_counter = 0
        
        # Initialize with basic instructions
        self._initialize_base_instructions()
    
    def _initialize_base_instructions(self):
        """Initialize with basic TAgent instructions."""
        base_instructions = [
            RagDocument(
                id="base_planning",
                content="When planning, break down the goal into specific, actionable tasks. Consider what tools are available and what data you need to collect.",
                doc_type="instruction",
                keywords=["plan", "planning", "strategy", "tasks"],
                goal_context="general"
            ),
            RagDocument(
                id="base_execution",
                content="When executing tools, use clear parameters and handle results systematically. Store important findings in the state for future reference.",
                doc_type="instruction", 
                keywords=["execute", "tool", "action", "parameters"],
                goal_context="general"
            ),
            RagDocument(
                id="base_evaluation",
                content="When evaluating goal achievement, be specific about what has been accomplished and what is still missing. Provide clear feedback for next steps.",
                doc_type="instruction",
                keywords=["evaluate", "assessment", "goal", "completion"],
                goal_context="general"
            )
        ]
        
        for doc in base_instructions:
            self.documents[doc.id] = doc
    
    def store_memories(self, memories: List[MemoryItem], action_context: str = ""):
        """Store memories from agent responses."""
        for memory in memories:
            doc_id = f"memory_{self.doc_counter}"
            self.doc_counter += 1
            
            # Extract keywords from memory content and relevance
            keywords = []
            if memory.relevance:
                keywords.extend(memory.relevance.split())
            
            # Add keywords based on memory type
            keywords.extend([memory.type, action_context])
            
            # Simple keyword extraction from content
            content_words = memory.content.lower().split()
            keywords.extend([word for word in content_words if len(word) > 3])
            
            doc = RagDocument(
                id=doc_id,
                content=memory.content,
                doc_type="memory",
                keywords=keywords,
                goal_context=self.goal,
                metadata={
                    "memory_type": memory.type,
                    "action_context": action_context,
                    "relevance": memory.relevance
                }
            )
            
            self.documents[doc_id] = doc
    
    def get_context_for_planning(self, current_state: Dict[str, Any]) -> str:
        """Get relevant context for planning actions."""
        # Note: current_state could be used for more sophisticated context extraction
        _ = current_state  # Mark as used for now
        query_keywords = ["plan", "planning", "strategy", "tasks"]
        
        # Add goal-specific keywords
        goal_words = self.goal.lower().split()
        query_keywords.extend([word for word in goal_words if len(word) > 3])
        
        relevant_docs = self._search_documents(query_keywords, doc_types=["instruction", "memory", "strategy"])
        
        if not relevant_docs:
            return ""
        
        context_parts = ["=== RELEVANT PLANNING CONTEXT ==="]
        for doc in relevant_docs[:5]:  # Limit to 5 most relevant
            context_parts.append(f"• {doc.content}")
        
        return "\n".join(context_parts)
    
    def get_context_for_execution(self, tool_name: str, task_description: str) -> str:
        """Get relevant context for execution actions."""
        query_keywords = ["execute", "tool", tool_name, "action"]
        
        # Add task-specific keywords
        task_words = task_description.lower().split()
        query_keywords.extend([word for word in task_words if len(word) > 3])
        
        relevant_docs = self._search_documents(query_keywords, doc_types=["instruction", "memory", "example"])
        
        if not relevant_docs:
            return ""
        
        context_parts = ["=== RELEVANT EXECUTION CONTEXT ==="]
        for doc in relevant_docs[:3]:  # Limit to 3 most relevant
            context_parts.append(f"• {doc.content}")
        
        return "\n".join(context_parts)
    
    def get_context_for_evaluation(self, current_state: Dict[str, Any]) -> str:
        """Get relevant context for evaluation actions."""
        # Note: current_state could be used for more sophisticated context extraction
        _ = current_state  # Mark as used for now
        query_keywords = ["evaluate", "assessment", "goal", "completion"]
        
        # Add goal-specific keywords
        goal_words = self.goal.lower().split()
        query_keywords.extend([word for word in goal_words if len(word) > 3])
        
        relevant_docs = self._search_documents(query_keywords, doc_types=["instruction", "memory", "strategy"])
        
        if not relevant_docs:
            return ""
        
        context_parts = ["=== RELEVANT EVALUATION CONTEXT ==="]
        for doc in relevant_docs[:3]:  # Limit to 3 most relevant
            context_parts.append(f"• {doc.content}")
        
        return "\n".join(context_parts)
    
    def _search_documents(self, query_keywords: List[str], doc_types: List[str] = None) -> List[RagDocument]:
        """Simple keyword-based document search."""
        if doc_types is None:
            doc_types = ["instruction", "memory", "strategy", "example"]
        
        scored_docs = []
        
        for doc in self.documents.values():
            if doc.doc_type not in doc_types:
                continue
            
            # Simple scoring based on keyword matches
            score = 0
            for keyword in query_keywords:
                keyword_lower = keyword.lower()
                
                # Score based on keyword presence
                if keyword_lower in doc.content.lower():
                    score += 2
                
                # Score based on keyword matches
                for doc_keyword in doc.keywords:
                    if keyword_lower in doc_keyword.lower():
                        score += 1
            
            # Boost score for goal-related documents
            if doc.goal_context and doc.goal_context in self.goal.lower():
                score += 1
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score (descending) and return documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        memory_docs = [doc for doc in self.documents.values() if doc.doc_type == "memory"]
        
        type_counts = {}
        for doc in memory_docs:
            mem_type = doc.metadata.get("memory_type", "unknown")
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "memory_documents": len(memory_docs),
            "memory_types": type_counts,
            "goal_context": self.goal
        }
    
    def clear_memories(self):
        """Clear all stored memories (keep base instructions)."""
        self.documents = {
            doc_id: doc for doc_id, doc in self.documents.items() 
            if doc.doc_type == "instruction" and doc.goal_context == "general"
        }
        self.doc_counter = 0


# Enhanced Context Manager is now an alias for the real RAG implementation
EnhancedContextManager = TFIDFRagContextManager
"""
Real RAG implementation with TF-IDF vectorization and cosine similarity.
This replaces the simple keyword-based context manager with actual semantic search.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import MemoryItem, EnhancedAgentState


@dataclass
class RagDocument:
    """Represents a document stored in RAG with embeddings."""
    
    id: str
    content: str
    doc_type: str  # 'instruction', 'memory', 'strategy', 'example'
    keywords: List[str]
    goal_context: Optional[str] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


class TFIDFRagContextManager:
    """
    Real RAG Context Manager using TF-IDF vectorization and cosine similarity.
    
    This implementation provides actual semantic search capabilities using:
    - TF-IDF vectorization for document representation
    - Cosine similarity for relevance scoring
    - Efficient document retrieval based on query similarity
    """
    
    def __init__(self, goal: str, max_features: int = 5000, min_df: int = 1, max_df: float = 1.0):
        self.goal = goal
        self.documents: Dict[str, RagDocument] = {}
        self.doc_counter = 0
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.document_matrix = None
        self.fitted = False
    
    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on all documents."""
        if not self.documents:
            return
            
        # Collect all document texts
        texts = [doc.content for doc in self.documents.values()]
        
        # Fit vectorizer and transform documents
        self.document_matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
    
    def _update_embeddings(self):
        """Update embeddings for all documents."""
        if not self.fitted:
            self._fit_vectorizer()
        elif self.document_matrix is not None:
            # Re-fit with new documents
            texts = [doc.content for doc in self.documents.values()]
            self.document_matrix = self.vectorizer.fit_transform(texts)
    
    def store_memories(self, memories: List[MemoryItem], action_context: str = ""):
        """Store memories from agent responses with TF-IDF embeddings."""
        for memory in memories:
            doc_id = f"memory_{self.doc_counter}"
            self.doc_counter += 1
            
            # Extract keywords from memory content and relevance
            keywords = []
            if memory.relevance:
                keywords.extend(memory.relevance.split())
            
            # Add keywords based on memory type
            keywords.extend([memory.type, action_context])
            
            # Enhanced keyword extraction
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
        
        # Update embeddings after adding new documents
        self._update_embeddings()
    
    def semantic_search(self, query: str, top_k: int = 5, doc_types: List[str] = None) -> List[Tuple[RagDocument, float]]:
        """
        Perform semantic search using TF-IDF and cosine similarity.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            doc_types: Filter by document types
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.fitted or self.document_matrix is None:
            self._fit_vectorizer()
        
        if not self.fitted:
            return []
        
        # Transform query using fitted vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.document_matrix).flatten()
        
        # Filter documents by type if specified
        filtered_docs = []
        for i, doc in enumerate(self.documents.values()):
            if doc_types is None or doc.doc_type in doc_types:
                filtered_docs.append((i, doc, similarities[i]))
        
        # Sort by similarity score
        filtered_docs.sort(key=lambda x: x[2], reverse=True)
        
        # Return top-k results
        return [(doc, score) for _, doc, score in filtered_docs[:top_k]]
    
    def get_context_for_current_state(self, state: EnhancedAgentState) -> str:
        """Get context based on the current enhanced agent state using semantic search."""
        # Build query based on current situation
        query_parts = [state.current_phase, state.goal]
        
        # Add failure context if available
        if failure_context := state.get_failure_context():
            query_parts.append(failure_context)
        
        # Add available tools context
        if state.available_tools:
            query_parts.extend(state.available_tools)
        
        # Add collected data context
        if state.collected_data:
            query_parts.extend(state.collected_data.keys())
        
        # Create search query
        query = " ".join(query_parts)
        
        # Perform semantic search
        results = self.semantic_search(query, top_k=5, doc_types=["instruction", "memory", "strategy"])
        
        if not results:
            return self._get_fallback_context(state.current_phase)
        
        # Build context response
        context_parts = [f"=== SEMANTIC CONTEXT FOR {state.current_phase.upper()} ==="]
        
        # Add failure context if relevant
        if failure_context := state.get_failure_context():
            context_parts.append(f"PREVIOUS FAILURE: {failure_context}")
        
        # Add semantically relevant documents
        for doc, score in results:
            context_parts.append(f"• [{score:.3f}] {doc.content}")
        
        # Add state-specific guidance
        if state.current_phase == "plan":
            context_parts.append(self._get_planning_guidance(state))
        elif state.current_phase == "execute":
            context_parts.append(self._get_execution_guidance(state))
        elif state.current_phase == "evaluate":
            context_parts.append(self._get_evaluation_guidance(state))
        
        return "\n".join(context_parts)
    
    def get_context_for_planning(self, current_state: Dict[str, Any]) -> str:
        """Get relevant context for planning actions using semantic search."""
        # Build planning-specific query
        goal_words = self.goal.lower().split()
        query = f"plan planning strategy tasks {' '.join([word for word in goal_words if len(word) > 3])}"
        
        # Perform semantic search
        results = self.semantic_search(query, top_k=5, doc_types=["instruction", "memory", "strategy"])
        
        if not results:
            return ""
        
        context_parts = ["=== SEMANTIC PLANNING CONTEXT ==="]
        for doc, score in results:
            context_parts.append(f"• [{score:.3f}] {doc.content}")
        
        return "\n".join(context_parts)
    
    def get_context_for_execution(self, tool_name: str, task_description: str) -> str:
        """Get relevant context for execution actions using semantic search."""
        # Build execution-specific query
        query = f"execute tool action {tool_name} {task_description}"
        
        # Perform semantic search
        results = self.semantic_search(query, top_k=3, doc_types=["instruction", "memory", "example"])
        
        if not results:
            return ""
        
        context_parts = ["=== SEMANTIC EXECUTION CONTEXT ==="]
        for doc, score in results:
            context_parts.append(f"• [{score:.3f}] {doc.content}")
        
        return "\n".join(context_parts)
    
    def get_context_for_evaluation(self, current_state: Dict[str, Any]) -> str:
        """Get relevant context for evaluation actions using semantic search."""
        # Build evaluation-specific query
        goal_words = self.goal.lower().split()
        query = f"evaluate assessment goal completion {' '.join([word for word in goal_words if len(word) > 3])}"
        
        # Perform semantic search
        results = self.semantic_search(query, top_k=3, doc_types=["instruction", "memory", "strategy"])
        
        if not results:
            return ""
        
        context_parts = ["=== SEMANTIC EVALUATION CONTEXT ==="]
        for doc, score in results:
            context_parts.append(f"• [{score:.3f}] {doc.content}")
        
        return "\n".join(context_parts)
    
    def _get_planning_guidance(self, state: EnhancedAgentState) -> str:
        """Get specific guidance for planning phase."""
        guidance = []
        
        if state.failure_reason:
            guidance.append(f"FOCUS: Address the issue - {state.failure_reason}")
        
        if state.available_tools:
            guidance.append(f"AVAILABLE TOOLS: {', '.join(state.available_tools)}")
        
        if state.collected_data:
            guidance.append(f"EXISTING DATA: {', '.join(state.collected_data.keys())}")
        
        return "\n".join(guidance) if guidance else ""
    
    def _get_execution_guidance(self, state: EnhancedAgentState) -> str:
        """Get specific guidance for execution phase."""
        guidance = []
        
        if state.available_tools:
            guidance.append(f"CHOOSE FROM TOOLS: {', '.join(state.available_tools)}")
        
        if state.failure_reason and "missing" in state.failure_reason.lower():
            guidance.append("FOCUS: Collect the missing data mentioned in failure reason")
        
        return "\n".join(guidance) if guidance else ""
    
    def _get_evaluation_guidance(self, state: EnhancedAgentState) -> str:
        """Get specific guidance for evaluation phase."""
        guidance = []
        
        guidance.append(f"EVALUATE GOAL: {state.goal}")
        
        if state.collected_data:
            guidance.append(f"EVALUATE DATA: {', '.join(state.collected_data.keys())}")
        
        guidance.append("BE SPECIFIC: If goal not achieved, provide clear failure reason")
        
        return "\n".join(guidance) if guidance else ""
    
    def _get_fallback_context(self, phase: str) -> str:
        """Get fallback context when no specific context is found."""
        fallbacks = {
            "plan": "Create a clear plan with specific tasks to achieve the goal.",
            "execute": "Execute the most appropriate tool with proper parameters.",
            "evaluate": "Evaluate if the goal has been achieved based on collected data."
        }
        return fallbacks.get(phase, "Proceed with the current phase.")
    
    def store_execution_result(self, state: EnhancedAgentState, action: str, result: Any, success: bool):
        """Store execution result as context for future retrieval."""
        result_type = "success" if success else "failure"
        
        # Create memory item from execution result
        memory_content = f"{action} {result_type}"
        if not success and state.failure_reason:
            memory_content += f": {state.failure_reason}"
        
        memory_item = MemoryItem(
            content=memory_content,
            type="execution_result",
            relevance=f"{action} {state.current_phase} {result_type}"
        )
        
        self.store_memories([memory_item], f"{action}_execution")
        
        # Add to state context history
        state.add_context_event(
            event_type="execution",
            description=f"{action} {result_type}",
            metadata={"success": success, "failure_reason": state.failure_reason}
        )
    
    def get_all_memories_for_finalize(self) -> List[RagDocument]:
        """Get all memories for finalize phase - comprehensive data retrieval."""
        memories = [doc for doc in self.documents.values() if doc.doc_type == "memory"]
        
        # Sort by timestamp (most recent first)
        memories.sort(key=lambda x: x.timestamp, reverse=True)
        
        return memories
    
    def get_memory_summary_for_finalize(self) -> Dict[str, Any]:
        """Get structured memory summary for finalize phase."""
        memories = self.get_all_memories_for_finalize()
        
        summary = {
            "total_memories": len(memories),
            "memory_types": {},
            "execution_results": [],
            "key_facts": [],
            "strategies": [],
            "lessons": [],
            "semantic_search_enabled": True,
            "vectorizer_features": self.vectorizer.get_feature_names_out().tolist() if self.fitted else []
        }
        
        for memory in memories:
            mem_type = memory.metadata.get("memory_type", "unknown")
            summary["memory_types"][mem_type] = summary["memory_types"].get(mem_type, 0) + 1
            
            # Categorize memories for better finalization
            if mem_type == "fact":
                summary["key_facts"].append(memory.content)
            elif mem_type == "strategy":
                summary["strategies"].append(memory.content)
            elif mem_type == "lesson":
                summary["lessons"].append(memory.content)
            elif memory.metadata.get("action_context"):
                summary["execution_results"].append({
                    "action": memory.metadata.get("action_context"),
                    "content": memory.content,
                    "timestamp": memory.timestamp
                })
        
        return summary
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories and RAG system."""
        memory_docs = [doc for doc in self.documents.values() if doc.doc_type == "memory"]
        
        type_counts = {}
        for doc in memory_docs:
            mem_type = doc.metadata.get("memory_type", "unknown")
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "memory_documents": len(memory_docs),
            "memory_types": type_counts,
            "goal_context": self.goal,
            "vectorizer_fitted": self.fitted,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.fitted else 0,
            "feature_count": self.vectorizer.max_features if self.fitted else 0
        }
    
    def clear_memories(self):
        """Clear all stored memories (keep base instructions)."""
        self.documents = {
            doc_id: doc for doc_id, doc in self.documents.items() 
            if doc.doc_type == "instruction" and doc.goal_context == "general"
        }
        self.doc_counter = 0
        
        # Re-fit vectorizer with remaining documents
        self._fit_vectorizer()
    
    def find_similar_documents(self, reference_doc: RagDocument, top_k: int = 3) -> List[Tuple[RagDocument, float]]:
        """Find documents similar to a reference document."""
        return self.semantic_search(reference_doc.content, top_k=top_k)
    
    def get_document_by_id(self, doc_id: str) -> Optional[RagDocument]:
        """Get a specific document by ID."""
        return self.documents.get(doc_id)
    
    def get_documents_by_type(self, doc_type: str) -> List[RagDocument]:
        """Get all documents of a specific type."""
        return [doc for doc in self.documents.values() if doc.doc_type == doc_type]


# Alias for backward compatibility
EnhancedRagContextManager = TFIDFRagContextManager
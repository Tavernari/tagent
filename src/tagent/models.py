"""
Pydantic models for TAgent state management and structured outputs.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Represents agent state as a typed dictionary."""

    data: Dict[str, Any] = {}


class EnhancedAgentState(BaseModel):
    """Enhanced agent state with historical context for RAG."""
    
    goal: str = Field(description="The main objective for the agent")
    current_phase: str = Field(description="Current phase: init, plan, execute, evaluate, finalize")
    last_action: Optional[str] = Field(default=None, description="Last action performed")
    last_result: Optional[str] = Field(default=None, description="Result of last action: success, failed, partial")
    failure_reason: Optional[str] = Field(default=None, description="Reason for failure if last_result is failed")
    iteration: int = Field(default=0, description="Current iteration number")
    available_tools: List[str] = Field(default_factory=list, description="List of available tool names")
    collected_data: Dict[str, Any] = Field(default_factory=dict, description="Data collected during execution")
    context_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historical context for RAG retrieval")
    
    def to_rag_query_context(self) -> Dict[str, Any]:
        """Convert to context optimized for RAG queries."""
        return {
            "goal": self.goal,
            "current_phase": self.current_phase,
            "last_action": self.last_action,
            "last_result": self.last_result,
            "failure_reason": self.failure_reason,
            "iteration": self.iteration,
            "available_tools": self.available_tools,
            "has_data": bool(self.collected_data),
            "data_keys": list(self.collected_data.keys()) if self.collected_data else []
        }
    
    def add_context_event(self, event_type: str, description: str, metadata: Dict[str, Any] = None):
        """Add an event to context history."""
        event = {
            "type": event_type,
            "description": description,
            "timestamp": self.iteration,
            "metadata": metadata or {}
        }
        self.context_history.append(event)
    
    def get_failure_context(self) -> Optional[str]:
        """Get context about the last failure for planning."""
        if self.last_result == "failed" and self.failure_reason:
            return f"Last {self.last_action} failed because: {self.failure_reason}"
        return None


class TokenUsage(BaseModel):
    """Represents token usage statistics for LLM interactions."""
    
    input_tokens: int = Field(default=0, description="Number of input tokens used")
    output_tokens: int = Field(default=0, description="Number of output tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens used (input + output)")
    model: str = Field(description="Model used for this interaction")
    cost: float = Field(default=0.0, description="Cost in USD for this interaction")
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add token usage statistics together."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            model=self.model,  # Keep the model from self
            cost=self.cost + other.cost
        )


class TokenStats(BaseModel):
    """Aggregated token statistics by model."""
    
    by_model: Dict[str, TokenUsage] = Field(default_factory=dict, description="Token usage grouped by model")
    total: TokenUsage = Field(default_factory=lambda: TokenUsage(model="total", input_tokens=0, output_tokens=0, total_tokens=0, cost=0.0), description="Total token usage across all models")
    
    def add_usage(self, usage: TokenUsage) -> None:
        """Add token usage to the statistics."""
        if usage.model not in self.by_model:
            self.by_model[usage.model] = TokenUsage(model=usage.model, input_tokens=0, output_tokens=0, total_tokens=0, cost=0.0)
        
        self.by_model[usage.model] = self.by_model[usage.model] + usage
        self.total = self.total + usage
    
    def format_dimmed_summary(self, model_name: Optional[str] = None) -> str:
        """Format a dimmed summary for logging."""
        cost_str = f"${self.total.cost:.6f}" if self.total.cost > 0 else "$0.000000"
        model_str = f" | model: {model_name}" if model_name else ""
        return f"↗ {self.total.input_tokens} ↙ {self.total.output_tokens} (Σ {self.total.total_tokens} tokens | {cost_str}{model_str})"


class MemoryItem(BaseModel):
    """Represents a single memory item to be stored in RAG."""
    
    content: str = Field(description="The fact or knowledge to remember")
    type: str = Field(description="Type of memory: 'fact', 'pattern', 'strategy', 'lesson', 'context'")
    relevance: Optional[str] = Field(default=None, description="When this memory is relevant (keywords)")


class StructuredResponse(BaseModel):
    """Schema for structured outputs generated by LLMs."""

    model_config = {
        "populate_by_name": True
    }  # Allow both 'params' and 'parameters' field names

    action: str = Field(
        description=(
            "Action to be taken, must be one of: "
            "'plan', 'execute', 'summarize', 'evaluate'"
        )
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the action",
        alias="parameters",
    )
    reasoning: str = Field(default="", description="Reasoning for the action")
    memories: List[MemoryItem] = Field(
        default_factory=list,
        description="Facts or knowledge to remember for future iterations"
    )
    stats: Optional[TokenStats] = Field(
        default=None,
        description="Token usage statistics for this interaction"
    )

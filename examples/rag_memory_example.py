"""
Example demonstrating the new RAG memory system in TAgent.
"""

from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from src.tagent.agent import run_agent
from src.tagent.models import MemoryItem


def example_research_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Example tool that simulates research and creates memories.
    """
    topic = args.get("topic", "general")
    
    # Simulate research results
    research_data = {
        "topic": topic,
        "findings": [
            f"Key finding about {topic}",
            f"Important insight regarding {topic}",
            f"Research conclusion on {topic}"
        ],
        "sources": ["source1.com", "source2.com"],
        "timestamp": "2025-01-15"
    }
    
    # This tool would normally create memories through the LLM response
    # but for demonstration, we'll show how the memory system works
    
    return ("research_data", research_data)


def example_analysis_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Example tool that performs analysis on existing data.
    """
    data_key = args.get("data_key", "research_data")
    
    existing_data = state.get(data_key, {})
    
    analysis_result = {
        "analyzed_data": existing_data,
        "insights": [
            "Pattern detected in the data",
            "Trend identified across sources",
            "Correlation found between variables"
        ],
        "confidence": 0.85,
        "recommendations": ["Action 1", "Action 2", "Action 3"]
    }
    
    return ("analysis_result", analysis_result)


class ResearchReport(BaseModel):
    """Output format for research reports."""
    
    topic: str = Field(description="The research topic")
    key_findings: list = Field(description="Key findings from research")
    analysis: str = Field(description="Analysis summary")
    recommendations: list = Field(description="Recommended actions")
    confidence: float = Field(description="Confidence score")


def run_rag_memory_example():
    """
    Demonstrate the RAG memory system with a research scenario.
    """
    print("=" * 60)
    print("RAG MEMORY SYSTEM DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Define tools
    tools = {
        "research_tool": example_research_tool,
        "analysis_tool": example_analysis_tool
    }
    
    # Define goal
    goal = "Research artificial intelligence trends and provide analysis with actionable recommendations"
    
    # Run agent with memory system
    print("🤖 Running TAgent with RAG Memory System...")
    print(f"📋 Goal: {goal}")
    print()
    
    result = run_agent(
        goal=goal,
        model="gpt-4",  # You can change this to any supported model
        tools=tools,
        output_format=ResearchReport,
        verbose=True,  # Enable verbose mode to see memory operations
        max_iterations=10
    )
    
    print("\n" + "=" * 60)
    print("EXECUTION RESULTS")
    print("=" * 60)
    
    # Show final result
    print(f"📊 Status: {result['status']}")
    print(f"🔄 Iterations used: {result['iterations_used']}")
    print(f"⚡ Execute actions: {result['executes_used']}")
    print(f"🧠 Memory stats: {result['memory_stats']}")
    
    if result['result']:
        print("\n📋 FINAL REPORT:")
        print(f"   Topic: {result['result'].topic}")
        print(f"   Key Findings: {len(result['result'].key_findings)} items")
        print(f"   Confidence: {result['result'].confidence}")
        print(f"   Recommendations: {len(result['result'].recommendations)} items")
    
    print("\n" + "=" * 60)
    print("MEMORY SYSTEM BENEFITS DEMONSTRATED:")
    print("=" * 60)
    print("✅ Memories stored during execution")
    print("✅ Context retrieved for each action")
    print("✅ Reduced conversation history overhead")
    print("✅ Incremental learning from each iteration")
    print("✅ Relevant context injection for better decisions")


if __name__ == "__main__":
    run_rag_memory_example()
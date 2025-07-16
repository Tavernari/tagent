"""
Example demonstrating the new RAG Linear Agent system.
Shows how the agent handles failures and uses RAG context for recovery.
"""

from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from src.tagent.rag_agent import run_rag_agent


def user_data_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Tool to collect user data. Simulates requiring specific fields.
    """
    required_fields = ["name", "email", "age"]
    provided_fields = args.get("fields", {})
    
    # Simulate missing data scenario
    missing_fields = [field for field in required_fields if field not in provided_fields]
    
    if missing_fields:
        return ("user_data_error", {
            "error": f"Missing required fields: {', '.join(missing_fields)}",
            "provided": provided_fields,
            "required": required_fields
        })
    
    # Simulate successful data collection
    user_data = {
        "user_id": "user_123",
        "name": provided_fields["name"],
        "email": provided_fields["email"],
        "age": provided_fields["age"],
        "status": "active",
        "created_at": "2025-01-15T10:30:00Z"
    }
    
    return ("user_data", user_data)


def validation_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Tool to validate collected data.
    """
    data_to_validate = args.get("data", {})
    
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Validate email format
    if "email" in data_to_validate:
        email = data_to_validate["email"]
        if "@" not in email:
            validation_results["valid"] = False
            validation_results["errors"].append("Invalid email format")
    
    # Validate age
    if "age" in data_to_validate:
        age = data_to_validate["age"]
        if not isinstance(age, int) or age < 0 or age > 150:
            validation_results["valid"] = False
            validation_results["errors"].append("Invalid age value")
    
    # Add success message if valid
    if validation_results["valid"]:
        validation_results["message"] = "All data is valid"
    
    return ("validation_result", validation_results)


def storage_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Tool to store validated data.
    """
    data_to_store = args.get("data", {})
    
    # Simulate storage process
    storage_result = {
        "stored": True,
        "storage_id": "storage_456",
        "timestamp": "2025-01-15T10:35:00Z",
        "data": data_to_store
    }
    
    return ("storage_result", storage_result)


class UserProcessingReport(BaseModel):
    """Final output format for user processing."""
    
    user_id: str = Field(description="Unique user identifier")
    name: str = Field(description="User's name")
    email: str = Field(description="User's email address")
    age: int = Field(description="User's age")
    processing_status: str = Field(description="Status of processing")
    validation_passed: bool = Field(description="Whether validation passed")
    storage_id: str = Field(description="Storage identifier")
    summary: str = Field(description="Summary of the processing")


def run_rag_linear_demo():
    """
    Demonstrate the RAG Linear Agent with failure recovery.
    """
    print("=" * 70)
    print("RAG LINEAR AGENT DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Define tools
    tools = {
        "user_data_tool": user_data_tool,
        "validation_tool": validation_tool,
        "storage_tool": storage_tool
    }
    
    # Test scenario: Goal that will initially fail due to missing data
    goal = "Collect user data including name, email, and age, validate it, and store it securely"
    
    print("🎯 GOAL:", goal)
    print()
    print("🔧 AVAILABLE TOOLS:")
    for tool_name, tool_func in tools.items():
        doc = tool_func.__doc__ or "No description"
        print(f"  - {tool_name}: {doc.strip()}")
    print()
    
    print("🚀 EXPECTED FLOW:")
    print("  1. PLAN: Create strategy to collect user data")
    print("  2. EXECUTE: Try to collect data (will fail - missing fields)")
    print("  3. EVALUATE: Detect failure due to missing data")
    print("  4. PLAN: Create new strategy addressing missing fields")
    print("  5. EXECUTE: Collect data with all required fields")
    print("  6. EVALUATE: Validate data collection")
    print("  7. EXECUTE: Validate collected data")
    print("  8. EXECUTE: Store validated data")
    print("  9. EVALUATE: Confirm goal achievement")
    print("  10. FINALIZE: Create comprehensive report")
    print()
    
    print("▶️  STARTING EXECUTION...")
    print("=" * 70)
    
    # Run the RAG agent
    result = run_rag_agent(
        goal=goal,
        tools=tools,
        output_format=UserProcessingReport,
        model="gpt-4",  # You can change this to any supported model
        verbose=True,
        max_iterations=10
    )
    
    print("\n" + "=" * 70)
    print("EXECUTION RESULTS")
    print("=" * 70)
    
    # Show comprehensive results
    print(f"🏆 Goal Achieved: {result.goal_achieved}")
    print(f"🔄 Iterations Used: {result.iterations_used}")
    print(f"⚡ Execute Count: {result.execute_count}")
    print(f"📋 Planning Count: {result.planning_count}")
    
    if result.failure_reason:
        print(f"❌ Final Failure: {result.failure_reason}")
    
    print(f"\n📊 PHASE HISTORY:")
    for i, phase in enumerate(result.phase_history, 1):
        print(f"  {i}. {phase}")
    
    print(f"\n🧠 MEMORY SUMMARY:")
    memory_summary = result.memory_summary
    print(f"  Total Memories: {memory_summary['total_memories']}")
    print(f"  Memory Types: {memory_summary['memory_types']}")
    
    if memory_summary['key_facts']:
        print(f"  Key Facts: {len(memory_summary['key_facts'])}")
        for fact in memory_summary['key_facts'][:3]:
            print(f"    - {fact}")
    
    if memory_summary['execution_results']:
        print(f"  Execution Results: {len(memory_summary['execution_results'])}")
        for result_item in memory_summary['execution_results'][:3]:
            print(f"    - {result_item['action']}: {result_item['content']}")
    
    # Show final output if available
    if result.final_output:
        print(f"\n📋 FINAL OUTPUT:")
        output = result.final_output
        print(f"  User ID: {output.user_id}")
        print(f"  Name: {output.name}")
        print(f"  Email: {output.email}")
        print(f"  Age: {output.age}")
        print(f"  Status: {output.processing_status}")
        print(f"  Validation: {'✅ Passed' if output.validation_passed else '❌ Failed'}")
        print(f"  Storage ID: {output.storage_id}")
        print(f"  Summary: {output.summary}")
    
    print("\n" + "=" * 70)
    print("RAG LINEAR SYSTEM BENEFITS DEMONSTRATED:")
    print("=" * 70)
    print("✅ Linear flow: INIT → PLAN → EXECUTE → EVALUATE → FINALIZE")
    print("✅ Failure recovery: Returns to PLAN with specific context")
    print("✅ RAG context: Dynamic prompts based on current state")
    print("✅ Memory accumulation: Each iteration adds to knowledge base")
    print("✅ Contextual planning: Addresses specific failure reasons")
    print("✅ Comprehensive finalization: Uses all collected memories")
    print("✅ No hardcoded instructions: Everything comes from RAG")
    print("✅ State preservation: Maintains context across iterations")
    
    return result


if __name__ == "__main__":
    run_rag_linear_demo()
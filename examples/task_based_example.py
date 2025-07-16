"""
Example demonstrating the new Task-Based Agent system.
Shows task execution with retry logic and failure recovery.
"""

from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from src.tagent.task_based_agent import run_task_based_agent


def data_collection_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Tool to collect data. Simulates different success/failure scenarios.
    """
    data_type = args.get("data_type", "general")
    required_params = args.get("required_params", [])
    
    # Simulate different scenarios based on data type
    if data_type == "user_profile":
        # This will fail first time due to missing email
        if "email" not in required_params:
            return ("data_collection_error", {
                "error": "Missing required parameter: email",
                "data_type": data_type,
                "required": ["name", "email", "age"]
            })
        
        # Success scenario
        return ("user_profile_data", {
            "user_id": "user_001",
            "name": required_params.get("name", "John Doe"),
            "email": required_params.get("email", "john@example.com"),
            "age": required_params.get("age", 30),
            "profile_complete": True
        })
    
    elif data_type == "preferences":
        # This will succeed immediately
        return ("preferences_data", {
            "theme": "dark",
            "language": "en",
            "notifications": True,
            "privacy_level": "medium"
        })
    
    elif data_type == "activity":
        # This will fail twice, then succeed
        retry_count = state.get("activity_retry_count", 0) + 1
        state["activity_retry_count"] = retry_count
        
        if retry_count < 3:
            return ("activity_error", {
                "error": f"Network timeout (attempt {retry_count})",
                "retry_count": retry_count
            })
        
        # Success on third try
        return ("activity_data", {
            "last_login": "2025-01-15T10:30:00Z",
            "sessions": 15,
            "total_time": "2h 45m",
            "retry_count": retry_count
        })
    
    return ("unknown_data", {"data_type": data_type})


def validation_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Tool to validate collected data.
    """
    data_key = args.get("data_key", "")
    
    if not data_key or data_key not in state:
        return ("validation_error", {
            "error": f"Data key '{data_key}' not found in state",
            "available_keys": list(state.keys())
        })
    
    data = state[data_key]
    
    # Simulate validation logic
    validation_result = {
        "data_key": data_key,
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check for error indicators
    if isinstance(data, dict) and "error" in data:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Data contains error: {data['error']}")
    
    # Check for required fields in user profile
    if data_key == "user_profile_data" and isinstance(data, dict):
        required_fields = ["user_id", "name", "email"]
        for field in required_fields:
            if field not in data:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
    
    return ("validation_result", validation_result)


def integration_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Tool to integrate validated data into final system.
    """
    data_keys = args.get("data_keys", [])
    
    integrated_data = {
        "integration_id": "int_001",
        "timestamp": "2025-01-15T10:45:00Z",
        "status": "success",
        "data_sources": []
    }
    
    # Collect data from specified keys
    for key in data_keys:
        if key in state:
            integrated_data["data_sources"].append({
                "key": key,
                "data": state[key]
            })
    
    # Simulate integration processing
    integrated_data["processed_records"] = len(integrated_data["data_sources"])
    integrated_data["processing_time"] = "0.5s"
    
    return ("integration_result", integrated_data)


class DataProcessingReport(BaseModel):
    """Final output format for data processing."""
    
    processing_id: str = Field(description="Unique processing identifier")
    goal_achieved: bool = Field(description="Whether the goal was achieved")
    tasks_completed: int = Field(description="Number of tasks completed")
    tasks_failed: int = Field(description="Number of tasks failed")
    data_collected: Dict[str, Any] = Field(description="Summary of collected data")
    validation_results: Dict[str, Any] = Field(description="Validation results")
    integration_status: str = Field(description="Integration status")
    retry_summary: Dict[str, Any] = Field(description="Summary of retry attempts")
    execution_summary: str = Field(description="Summary of the entire execution")


def run_task_based_demo():
    """
    Demonstrate the Task-Based Agent with retry logic and failure recovery.
    """
    print("=" * 70)
    print("TASK-BASED AGENT DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Define tools
    tools = {
        "data_collection_tool": data_collection_tool,
        "validation_tool": validation_tool,
        "integration_tool": integration_tool
    }
    
    # Test scenario with multiple failure points
    goal = "Collect user profile, preferences, and activity data, validate all data, and integrate into the system"
    
    print("🎯 GOAL:", goal)
    print()
    print("🔧 AVAILABLE TOOLS:")
    for tool_name, tool_func in tools.items():
        doc = tool_func.__doc__ or "No description"
        print(f"  - {tool_name}: {doc.strip()}")
    print()
    
    print("🚀 EXPECTED FLOW:")
    print("  1. PLAN: Create tasks for data collection, validation, and integration")
    print("  2. EXECUTE: Try to collect user profile (will fail - missing email)")
    print("  3. EXECUTE: Retry user profile collection (will succeed)")
    print("  4. EXECUTE: Collect preferences (will succeed)")
    print("  5. EXECUTE: Collect activity data (will fail twice, succeed on 3rd try)")
    print("  6. EXECUTE: Validate all collected data")
    print("  7. EXECUTE: Integrate validated data")
    print("  8. EVALUATE: Check if goal achieved")
    print("  9. FINALIZE: Create comprehensive report")
    print()
    
    print("💡 RETRY LOGIC:")
    print("  - Each task can retry up to 3 times")
    print("  - After 3 failures, task marked as failed")
    print("  - If too many tasks fail, returns to PLAN with failure context")
    print()
    
    print("▶️  STARTING EXECUTION...")
    print("=" * 70)
    
    # Run the task-based agent
    result = run_task_based_agent(
        goal=goal,
        tools=tools,
        output_format=DataProcessingReport,
        model="gpt-4",  # You can change this to any supported model
        verbose=True,
        max_iterations=30,
        max_planning_cycles=3
    )
    
    print("\n" + "=" * 70)
    print("EXECUTION RESULTS")
    print("=" * 70)
    
    # Show comprehensive results
    print(f"🏆 Goal Achieved: {result.goal_achieved}")
    print(f"🔄 Iterations Used: {result.iterations_used}")
    print(f"📋 Planning Cycles: {result.planning_cycles}")
    print(f"📊 Tasks: {result.completed_tasks} completed, {result.failed_tasks} failed ({result.total_tasks} total)")
    
    if result.failure_reason:
        print(f"❌ Failure Reason: {result.failure_reason}")
    
    # Show task summary
    task_summary = result.state_summary.get('task_summary', {})
    print(f"\n📈 TASK BREAKDOWN:")
    print(f"  Total: {task_summary.get('total_tasks', 0)}")
    print(f"  Completed: {task_summary.get('completed', 0)}")
    print(f"  Failed: {task_summary.get('failed', 0)}")
    print(f"  Pending: {task_summary.get('pending', 0)}")
    print(f"  Retrying: {task_summary.get('retrying', 0)}")
    
    # Show memory summary
    memory_summary = result.memory_summary
    print(f"\n🧠 MEMORY SUMMARY:")
    print(f"  Total Memories: {memory_summary['total_memories']}")
    print(f"  Memory Types: {memory_summary['memory_types']}")
    
    if memory_summary['execution_results']:
        print(f"  Execution Results: {len(memory_summary['execution_results'])}")
        for i, result_item in enumerate(memory_summary['execution_results'][:5], 1):
            print(f"    {i}. {result_item['action']}: {result_item['content'][:50]}...")
    
    # Show final output if available
    if result.final_output:
        print(f"\n📋 FINAL OUTPUT:")
        output = result.final_output
        print(f"  Processing ID: {output.processing_id}")
        print(f"  Goal Achieved: {output.goal_achieved}")
        print(f"  Tasks Completed: {output.tasks_completed}")
        print(f"  Tasks Failed: {output.tasks_failed}")
        print(f"  Data Collected: {len(output.data_collected)} items")
        print(f"  Integration Status: {output.integration_status}")
        print(f"  Summary: {output.execution_summary[:100]}...")
    
    print("\n" + "=" * 70)
    print("TASK-BASED SYSTEM BENEFITS DEMONSTRATED:")
    print("=" * 70)
    print("✅ Task-oriented execution: Clear, discrete units of work")
    print("✅ Retry logic: Automatic retry up to 3 times per task")
    print("✅ Failure recovery: Returns to PLAN with specific failure context")
    print("✅ Progress tracking: Clear visibility into task completion")
    print("✅ RAG integration: Context-aware planning and execution")
    print("✅ Memory accumulation: Learning from each execution attempt")
    print("✅ Loop prevention: Max retry limits prevent infinite loops")
    print("✅ Comprehensive reporting: Full execution history and results")
    
    return result


if __name__ == "__main__":
    run_task_based_demo()
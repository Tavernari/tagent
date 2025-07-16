"""
Task-based State Machine for TAgent - Loop through tasks with retry logic.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from .models import EnhancedAgentState


class TaskStatus(Enum):
    """Status of individual tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class AgentPhase(Enum):
    """High-level phases of the agent."""
    INIT = "init"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a single task to be executed."""
    id: str
    description: str
    tool_args: Dict[str, Any]
    tool_name: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    failure_reason: Optional[str] = None
    result: Optional[Any] = None
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    def mark_failed(self, reason: str):
        """Mark task as failed with reason."""
        self.status = TaskStatus.FAILED
        self.failure_reason = reason
    
    def retry(self):
        """Increment retry count and mark as retrying."""
        self.retry_count += 1
        self.status = TaskStatus.RETRYING
    
    def mark_completed(self, result: Any):
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.result = result


class TaskBasedStateMachine:
    """
    Task-based state machine for TAgent.
    
    Flow:
    1. PLAN → generates list of tasks
    2. EXECUTE → loop through tasks:
       - Try to execute each task
       - If task fails, retry up to 3 times
       - If still fails, go back to PLAN with status
    3. EVALUATE → when all tasks completed/failed
    4. Based on evaluation: COMPLETED or back to PLAN
    """
    
    def __init__(self, goal: str, available_tools: List[str] = None):
        """Initialize the task-based state machine."""
        self.state = EnhancedAgentState(
            goal=goal,
            current_phase=AgentPhase.INIT.value,
            available_tools=available_tools or []
        )
        self.tasks: List[Task] = []
        self.current_task_index = 0
        self.max_iterations = 20
        self.current_iteration = 0
        self.planning_cycles = 0
        self.max_planning_cycles = 5
    
    def get_current_phase(self) -> AgentPhase:
        """Get current phase as enum."""
        return AgentPhase(self.state.current_phase)
    
    def set_phase(self, phase: AgentPhase):
        """Set current phase."""
        self.state.current_phase = phase.value
        self.state.iteration = self.current_iteration
    
    def add_tasks(self, tasks: List[Dict[str, Any]]):
        """Add tasks from planning phase."""
        self.tasks.clear()
        self.current_task_index = 0
        
        for i, task_data in enumerate(tasks):
            task = Task(
                id=f"task_{i+1}",
                description=task_data.get("description", ""),
                tool_name=task_data.get("tool_name"),
                tool_args=task_data.get("tool_args", {}),
                max_retries=task_data.get("max_retries", 3)
            )
            self.tasks.append(task)
    
    def get_current_task(self) -> Optional[Task]:
        """Get the current task being executed."""
        if 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next pending task."""
        for i in range(self.current_task_index, len(self.tasks)):
            task = self.tasks[i]
            if task.status in [TaskStatus.PENDING, TaskStatus.RETRYING]:
                self.current_task_index = i
                return task
        return None
    
    def complete_current_task(self, result: Any):
        """Mark current task as completed."""
        if current_task := self.get_current_task():
            current_task.mark_completed(result)
            
            # Add result to collected data
            if isinstance(result, tuple) and len(result) == 2:
                key, value = result
                self.state.collected_data[key] = value
            
            # Move to next task
            self.current_task_index += 1
    
    def fail_current_task(self, reason: str) -> bool:
        """
        Mark current task as failed and handle retry logic.
        
        Returns:
            True if task can be retried, False if max retries reached
        """
        if current_task := self.get_current_task():
            if current_task.can_retry():
                current_task.retry()
                return True
            else:
                current_task.mark_failed(reason)
                self.current_task_index += 1
                return False
        return False
    
    def all_tasks_processed(self) -> bool:
        """Check if all tasks have been processed (completed or failed)."""
        return all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            for task in self.tasks
        )
    
    def has_pending_tasks(self) -> bool:
        """Check if there are pending tasks to execute."""
        return any(
            task.status in [TaskStatus.PENDING, TaskStatus.RETRYING]
            for task in self.tasks
        )
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of task statuses."""
        summary = {
            "total_tasks": len(self.tasks),
            "completed": 0,
            "failed": 0,
            "pending": 0,
            "retrying": 0,
            "current_task_index": self.current_task_index
        }
        
        for task in self.tasks:
            if task.status == TaskStatus.COMPLETED:
                summary["completed"] += 1
            elif task.status == TaskStatus.FAILED:
                summary["failed"] += 1
            elif task.status == TaskStatus.PENDING:
                summary["pending"] += 1
            elif task.status == TaskStatus.RETRYING:
                summary["retrying"] += 1
        
        return summary
    
    def get_failed_tasks(self) -> List[Task]:
        """Get list of failed tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.FAILED]
    
    def get_failed_tasks_context(self) -> str:
        """Get context about failed tasks for planning."""
        failed_tasks = self.get_failed_tasks()
        
        if not failed_tasks:
            return ""
        
        context_parts = ["FAILED TASKS:"]
        for task in failed_tasks:
            context_parts.append(f"- {task.description}: {task.failure_reason}")
        
        return "\n".join(context_parts)
    
    def should_return_to_planning(self) -> bool:
        """Check if should return to planning due to failed tasks."""
        failed_tasks = self.get_failed_tasks()
        
        # Return to planning if:
        # 1. There are failed tasks AND
        # 2. We haven't exceeded max planning cycles
        return (
            len(failed_tasks) > 0 
            and self.planning_cycles < self.max_planning_cycles
        )
    
    def start_new_planning_cycle(self):
        """Start a new planning cycle."""
        self.planning_cycles += 1
        self.current_iteration += 1
        self.set_phase(AgentPhase.PLANNING)
        
        # Update state with failure context
        failed_context = self.get_failed_tasks_context()
        self.state.failure_reason = failed_context if failed_context else None
        self.state.last_result = "failed" if failed_context else "success"
    
    def should_continue(self) -> bool:
        """Check if agent should continue executing."""
        return (
            self.current_iteration < self.max_iterations
            and self.planning_cycles < self.max_planning_cycles
            and self.get_current_phase() not in [AgentPhase.COMPLETED, AgentPhase.FAILED]
        )
    
    def get_execution_context(self) -> Dict[str, Any]:
        """Get context for current execution."""
        current_task = self.get_current_task()
        
        return {
            "current_phase": self.state.current_phase,
            "current_task": {
                "id": current_task.id if current_task else None,
                "description": current_task.description if current_task else None,
                "tool_name": current_task.tool_name if current_task else None,
                "retry_count": current_task.retry_count if current_task else 0,
                "status": current_task.status.value if current_task else None
            },
            "task_summary": self.get_task_summary(),
            "iteration": self.current_iteration,
            "planning_cycles": self.planning_cycles,
            "collected_data_keys": list(self.state.collected_data.keys()),
            "failure_reason": self.state.failure_reason
        }
    
    def get_planning_context(self) -> Dict[str, Any]:
        """Get context for planning phase."""
        return {
            "goal": self.state.goal,
            "planning_cycle": self.planning_cycles,
            "previous_tasks": [
                {
                    "description": task.description,
                    "status": task.status.value,
                    "failure_reason": task.failure_reason
                }
                for task in self.tasks
            ],
            "failed_tasks_context": self.get_failed_tasks_context(),
            "available_tools": self.state.available_tools,
            "collected_data": self.state.collected_data
        }
    
    def get_evaluation_context(self) -> Dict[str, Any]:
        """Get context for evaluation phase."""
        task_summary = self.get_task_summary()
        
        return {
            "goal": self.state.goal,
            "task_summary": task_summary,
            "completed_tasks": task_summary["completed"],
            "failed_tasks": task_summary["failed"],
            "total_tasks": task_summary["total_tasks"],
            "collected_data": self.state.collected_data,
            "failed_tasks_details": [
                {
                    "description": task.description,
                    "failure_reason": task.failure_reason,
                    "retry_count": task.retry_count
                }
                for task in self.get_failed_tasks()
            ]
        }
    
    def mark_completed(self, success: bool = True):
        """Mark the entire process as completed."""
        if success:
            self.set_phase(AgentPhase.COMPLETED)
        else:
            self.set_phase(AgentPhase.FAILED)
            
        self.state.last_result = "success" if success else "failed"
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary."""
        return {
            "current_phase": self.state.current_phase,
            "goal": self.state.goal,
            "iteration": self.current_iteration,
            "planning_cycles": self.planning_cycles,
            "max_iterations": self.max_iterations,
            "max_planning_cycles": self.max_planning_cycles,
            "task_summary": self.get_task_summary(),
            "failed_tasks_count": len(self.get_failed_tasks()),
            "collected_data_count": len(self.state.collected_data),
            "should_continue": self.should_continue(),
            "failure_reason": self.state.failure_reason
        }
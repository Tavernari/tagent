"""
Pipeline State Management for TAgent Pipeline System.

This module extends the existing TaskBasedStateMachine to support pipeline execution
with memory persistence, shared context management, and step progress tracking.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from ..state_machine import TaskBasedStateMachine
from .models import Pipeline, PipelineStep, PipelineResult, StepStatus, PipelineStepContext
from .scheduler import PipelineScheduler


class PipelinePhase(Enum):
    """Extended phases for pipeline execution."""
    INIT = "init"
    PLANNING = "planning"
    SCHEDULING = "scheduling"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    RESUMING = "resuming"


@dataclass
class StepMemoryEntry:
    """Memory entry for a single step."""
    step_name: str
    pipeline_id: str
    data: Any
    timestamp: datetime
    execution_id: str
    dependencies_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default execution_id if not provided."""
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())


class PipelineMemory:
    """Enhanced memory management for pipeline execution."""
    
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.step_results: Dict[str, StepMemoryEntry] = {}
        self.shared_data: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.current_step: Optional[str] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
    def save_step_result(self, step_name: str, result: Any, dependencies_used: List[str] = None):
        """Save step result with timestamp and metadata."""
        entry = StepMemoryEntry(
            step_name=step_name,
            pipeline_id=self.pipeline_id,
            data=result,
            timestamp=datetime.now(),
            execution_id=str(uuid.uuid4()),
            dependencies_used=dependencies_used or [],
            metadata={}
        )
        
        self.step_results[step_name] = entry
        self.updated_at = datetime.now()
        self._update_execution_history(step_name, "completed", result)
    
    def get_step_result(self, step_name: str) -> Any:
        """Get result from specific step."""
        entry = self.step_results.get(step_name)
        return entry.data if entry else None
    
    def has_step_result(self, step_name: str) -> bool:
        """Check if step has completed with result."""
        return step_name in self.step_results
    
    def get_step_context(self, step_name: str) -> Dict[str, Any]:
        """Get execution context for a step."""
        return {
            "step_name": step_name,
            "pipeline_id": self.pipeline_id,
            "shared_data": self.shared_data.copy(),
            "execution_history": self.execution_history.copy(),
            "current_step": self.current_step,
            "timestamp": datetime.now()
        }
    
    def update_shared_data(self, key: str, value: Any):
        """Update shared data accessible by all steps."""
        self.shared_data[key] = value
        self.updated_at = datetime.now()
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get shared data by key."""
        return self.shared_data.get(key, default)
    
    def clear_step_result(self, step_name: str):
        """Clear result for a specific step."""
        if step_name in self.step_results:
            del self.step_results[step_name]
            self.updated_at = datetime.now()
            self._update_execution_history(step_name, "cleared", None)
    
    def get_dependency_results(self, dependencies: List[str]) -> Dict[str, Any]:
        """Get results from dependency steps."""
        results = {}
        for dep in dependencies:
            result = self.get_step_result(dep)
            if result is not None:
                results[dep] = result
        return results
    
    def _update_execution_history(self, step_name: str, event_type: str, data: Any):
        """Update execution history with event."""
        event = {
            "timestamp": datetime.now(),
            "step_name": step_name,
            "event_type": event_type,
            "data": data,
            "execution_id": str(uuid.uuid4())
        }
        self.execution_history.append(event)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of memory state."""
        return {
            "pipeline_id": self.pipeline_id,
            "step_results_count": len(self.step_results),
            "shared_data_keys": list(self.shared_data.keys()),
            "execution_history_count": len(self.execution_history),
            "current_step": self.current_step,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_steps": list(self.step_results.keys())
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize memory state for persistence."""
        return {
            "pipeline_id": self.pipeline_id,
            "step_results": {
                name: {
                    "step_name": entry.step_name,
                    "pipeline_id": entry.pipeline_id,
                    "data": entry.data,
                    "timestamp": entry.timestamp.isoformat(),
                    "execution_id": entry.execution_id,
                    "dependencies_used": entry.dependencies_used,
                    "metadata": entry.metadata
                }
                for name, entry in self.step_results.items()
            },
            "shared_data": self.shared_data,
            "execution_history": [
                {
                    **event,
                    "timestamp": event["timestamp"].isoformat()
                }
                for event in self.execution_history
            ],
            "current_step": self.current_step,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'PipelineMemory':
        """Deserialize memory state from persistence."""
        memory = cls(data["pipeline_id"])
        
        # Restore step results
        for name, entry_data in data.get("step_results", {}).items():
            entry = StepMemoryEntry(
                step_name=entry_data["step_name"],
                pipeline_id=entry_data["pipeline_id"],
                data=entry_data["data"],
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                execution_id=entry_data["execution_id"],
                dependencies_used=entry_data.get("dependencies_used", []),
                metadata=entry_data.get("metadata", {})
            )
            memory.step_results[name] = entry
        
        # Restore other data
        memory.shared_data = data.get("shared_data", {})
        memory.current_step = data.get("current_step")
        memory.created_at = datetime.fromisoformat(data["created_at"])
        memory.updated_at = datetime.fromisoformat(data["updated_at"])
        
        # Restore execution history
        memory.execution_history = [
            {
                **event,
                "timestamp": datetime.fromisoformat(event["timestamp"])
            }
            for event in data.get("execution_history", [])
        ]
        
        return memory


class PipelineStateMachine(TaskBasedStateMachine):
    """Enhanced state machine for pipeline execution."""
    
    def __init__(self, pipeline: Pipeline, memory_manager: Optional['PipelineMemoryManager'] = None):
        super().__init__(pipeline.description, [])
        self.pipeline = pipeline
        self.memory_manager = memory_manager
        self.pipeline_memory = PipelineMemory(pipeline.pipeline_id)
        self.scheduler = PipelineScheduler(pipeline)
        self.current_phase = PipelinePhase.INIT
        self.step_execution_times: Dict[str, float] = {}
        self.step_retry_counts: Dict[str, int] = {}
        self.paused_at: Optional[datetime] = None
        self.resumed_at: Optional[datetime] = None
        
        # Initialize step retry counts
        for step in pipeline.steps:
            self.step_retry_counts[step.name] = 0
    
    def get_current_phase(self) -> PipelinePhase:
        """Get current phase as enum."""
        return self.current_phase
    
    def set_phase(self, phase: PipelinePhase):
        """Set current phase."""
        self.current_phase = phase
        self.state.current_phase = phase.value
        self.state.iteration = self.current_iteration
    
    def get_ready_steps(self) -> List[PipelineStep]:
        """Get steps ready for execution based on dependencies."""
        ready_step_names = self.scheduler.get_ready_steps()
        ready_steps = []
        
        for step_name in ready_step_names:
            step = self.pipeline.get_step_by_name(step_name)
            if step and step.status == StepStatus.PENDING:
                ready_steps.append(step)
        
        return ready_steps
    
    def can_execute_step(self, step: PipelineStep) -> bool:
        """Check if step can execute based on dependencies."""
        # Check if step already completed
        if self.pipeline_memory.has_step_result(step.name):
            return False
        
        # Check if step is ready according to scheduler
        return self.scheduler.is_step_ready(step.name)
    
    def prepare_step_context(self, step: PipelineStep) -> PipelineStepContext:
        """Prepare execution context for a step."""
        dependencies = self.scheduler.get_step_dependencies(step.name)
        dependency_results = self.pipeline_memory.get_dependency_results(dependencies)
        
        context = PipelineStepContext(
            step_name=step.name,
            pipeline_name=self.pipeline.name,
            pipeline_id=self.pipeline.pipeline_id,
            step_dependencies=dependencies,
            dependency_results=dependency_results,
            shared_context=self.pipeline_memory.shared_data.copy(),
            execution_metadata={
                "current_phase": self.current_phase.value,
                "retry_count": self.step_retry_counts.get(step.name, 0),
                "execution_time": datetime.now().isoformat()
            },
            retry_count=self.step_retry_counts.get(step.name, 0),
            max_retries=step.max_retries
        )
        
        return context
    
    def start_step_execution(self, step: PipelineStep):
        """Mark step as running and update state."""
        step.mark_running()
        self.scheduler.update_step_status(step.name, StepStatus.RUNNING)
        self.pipeline_memory.current_step = step.name
        self.pipeline_memory.updated_at = datetime.now()
    
    def complete_step_execution(self, step: PipelineStep, result: Any):
        """Complete step execution with result."""
        step.mark_completed(result)
        self.scheduler.update_step_status(step.name, StepStatus.COMPLETED)
        
        # Save result to memory
        dependencies = self.scheduler.get_step_dependencies(step.name)
        self.pipeline_memory.save_step_result(step.name, result, dependencies)
        
        # Update execution time
        if step.get_execution_duration():
            self.step_execution_times[step.name] = step.get_execution_duration()
        
        # Add result to collected data (for compatibility with base class)
        if isinstance(result, tuple) and len(result) == 2:
            key, value = result
            self.state.collected_data[key] = value
    
    def fail_step_execution(self, step: PipelineStep, error_message: str) -> bool:
        """Fail step execution and handle retry logic."""
        # Try to retry first
        if step.retry():
            self.step_retry_counts[step.name] += 1
            self.scheduler.update_step_status(step.name, StepStatus.PENDING)
            return True
        else:
            # Max retries reached
            step.mark_failed(error_message)
            self.scheduler.update_step_status(step.name, StepStatus.FAILED)
            return False
    
    def get_execution_progress(self) -> Dict[str, Any]:
        """Get current execution progress."""
        total_steps = len(self.pipeline.steps)
        completed_steps = len(self.pipeline.get_steps_by_status(StepStatus.COMPLETED))
        failed_steps = len(self.pipeline.get_steps_by_status(StepStatus.FAILED))
        running_steps = len(self.pipeline.get_steps_by_status(StepStatus.RUNNING))
        pending_steps = len(self.pipeline.get_steps_by_status(StepStatus.PENDING))
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "running_steps": running_steps,
            "pending_steps": pending_steps,
            "progress_percentage": (completed_steps / total_steps) * 100 if total_steps > 0 else 0,
            "current_phase": self.current_phase.value,
            "ready_steps": [step.name for step in self.get_ready_steps()],
            "current_step": self.pipeline_memory.current_step
        }
    
    def is_pipeline_complete(self) -> bool:
        """Check if pipeline execution is complete."""
        return all(
            step.status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]
            for step in self.pipeline.steps
        )
    
    def has_pipeline_failed(self) -> bool:
        """Check if pipeline has failed (has failed steps with no retries)."""
        return any(
            step.status == StepStatus.FAILED and not step.can_retry()
            for step in self.pipeline.steps
        )
    
    def pause_execution(self):
        """Pause pipeline execution."""
        self.set_phase(PipelinePhase.PAUSED)
        self.paused_at = datetime.now()
    
    def resume_execution(self):
        """Resume pipeline execution."""
        self.set_phase(PipelinePhase.RESUMING)
        self.resumed_at = datetime.now()
    
    def get_failed_steps(self) -> List[PipelineStep]:
        """Get list of failed steps."""
        return self.pipeline.get_steps_by_status(StepStatus.FAILED)
    
    def get_completed_steps(self) -> List[PipelineStep]:
        """Get list of completed steps."""
        return self.pipeline.get_steps_by_status(StepStatus.COMPLETED)
    
    def create_pipeline_result(self, success: bool = None) -> PipelineResult:
        """Create final pipeline result."""
        if success is None:
            success = self.is_pipeline_complete() and not self.has_pipeline_failed()
        
        end_time = datetime.now()
        execution_time = (end_time - self.pipeline.created_at).total_seconds()
        
        result = PipelineResult(
            pipeline_name=self.pipeline.name,
            pipeline_id=self.pipeline.pipeline_id,
            success=success,
            execution_time=execution_time,
            start_time=self.pipeline.created_at,
            end_time=end_time,
            steps_completed=len(self.get_completed_steps()),
            steps_failed=len(self.get_failed_steps()),
            total_steps=len(self.pipeline.steps)
        )
        
        # Add step outputs and metadata
        for step in self.pipeline.steps:
            if step.status == StepStatus.COMPLETED and step.result:
                result.add_step_result(step.name, step.result, {
                    "execution_time": step.get_execution_duration(),
                    "retry_count": self.step_retry_counts.get(step.name, 0),
                    "end_time": step.end_time
                })
            elif step.status == StepStatus.FAILED:
                result.add_step_error(step.name, step.error_message or "Unknown error")
        
        # Add memory artifacts
        result.learned_facts = self.pipeline_memory.shared_data.copy()
        result.saved_memories = self.pipeline_memory.get_execution_summary()
        
        return result
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary."""
        base_summary = super().get_state_summary()
        
        pipeline_summary = {
            "pipeline_id": self.pipeline.pipeline_id,
            "pipeline_name": self.pipeline.name,
            "current_phase": self.current_phase.value,
            "execution_progress": self.get_execution_progress(),
            "memory_summary": self.pipeline_memory.get_execution_summary(),
            "scheduler_summary": self.scheduler.get_scheduling_summary(),
            "step_retry_counts": self.step_retry_counts,
            "step_execution_times": self.step_execution_times,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "resumed_at": self.resumed_at.isoformat() if self.resumed_at else None
        }
        
        return {**base_summary, **pipeline_summary}
    
    def persist_state(self):
        """Persist current state to memory manager."""
        if self.memory_manager:
            self.memory_manager.persist_memory(self.pipeline.pipeline_id, self.pipeline_memory)
    
    def restore_state(self):
        """Restore state from memory manager."""
        if self.memory_manager:
            restored_memory = self.memory_manager.restore_memory(self.pipeline.pipeline_id)
            if restored_memory:
                self.pipeline_memory = restored_memory
                
                # Update pipeline steps based on restored memory
                for step in self.pipeline.steps:
                    if self.pipeline_memory.has_step_result(step.name):
                        step.status = StepStatus.COMPLETED
                        step.result = self.pipeline_memory.get_step_result(step.name)
                        self.scheduler.update_step_status(step.name, StepStatus.COMPLETED)
                
                return True
        return False
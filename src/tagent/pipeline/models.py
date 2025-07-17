"""
Pipeline models for TAgent Pipeline System.

This module contains the foundational data models for pipeline definition and execution,
including pipeline steps, pipeline orchestration, and execution results.
"""

from typing import Dict, Any, List, Optional, Type, Union, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

from ..models import TokenUsage
from .conditions import AnyCondition

if TYPE_CHECKING:
    from ..config import TAgentConfig


class ExecutionMode(Enum):
    """Execution modes for pipeline steps."""
    SERIAL = "serial"
    CONCURRENT = "concurrent"


class DependencyType(Enum):
    """Types of dependencies between pipeline steps."""
    STRONG = "strong"  # Must complete successfully
    WEAK = "weak"      # Can fail but step can still run
    OPTIONAL = "optional"  # Dependency is optional


class StepStatus(Enum):
    """Status of individual pipeline steps."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """Individual step in a pipeline."""
    name: str
    goal: str
    constraints: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SERIAL
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    tools: Optional[List[Callable]] = None
    tools_filter: Optional[List[str]] = None
    output_schema: Optional[Type[BaseModel]] = None
    agent_config: Optional['TAgentConfig'] = None  # Step-specific TAgent configuration
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_id: Optional[str] = None
    condition: Optional[AnyCondition] = None
    condition_mode: str = "pre"  # "pre" or "post"
    
    def __post_init__(self):
        """Validate step configuration."""
        if not self.name:
            raise ValueError("Step name cannot be empty")
        if not self.goal:
            raise ValueError("Step goal cannot be empty")
        if self.execution_id is None:
            self.execution_id = str(uuid.uuid4())
    
    def validate_output(self, result: Any) -> Union[BaseModel, Any]:
        """Validate step output against schema."""
        if self.output_schema:
            if isinstance(result, dict):
                return self.output_schema(**result)
            elif isinstance(result, self.output_schema):
                return result
            else:
                raise ValueError(f"Step '{self.name}' output does not match schema {self.output_schema}")
        return result
    
    def can_retry(self) -> bool:
        """Check if step can be retried."""
        return self.retry_count < self.max_retries
    
    def mark_running(self):
        """Mark step as running."""
        self.status = StepStatus.RUNNING
        self.start_time = datetime.now()
    
    def mark_completed(self, result: Any):
        """Mark step as completed with result."""
        self.status = StepStatus.COMPLETED
        self.result = self.validate_output(result)
        self.end_time = datetime.now()
        self.error_message = None
    
    def mark_failed(self, error_message: str):
        """Mark step as failed with error message."""
        self.status = StepStatus.FAILED
        self.error_message = error_message
        self.end_time = datetime.now()
    
    def retry(self):
        """Increment retry count and reset for retry."""
        if self.can_retry():
            self.retry_count += 1
            self.status = StepStatus.PENDING
            self.start_time = None
            self.end_time = None
            self.error_message = None
            return True
        return False
    
    def get_execution_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def get_execution_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class PipelineSummary(BaseModel):
    """Summary of pipeline state."""
    pipeline_id: str = Field(description="Unique pipeline identifier")
    name: str = Field(description="Pipeline name")
    description: str = Field(description="Pipeline description")
    total_steps: int = Field(description="Total number of steps")
    status_counts: Dict[str, int] = Field(default_factory=dict, description="Count of steps by status")
    created_at: str = Field(description="Creation timestamp")
    global_constraints: List[str] = Field(default_factory=list, description="Global constraints")
    shared_context_keys: List[str] = Field(default_factory=list, description="Keys in shared context")


class SharedPipelineContext(BaseModel):
    """Shared context data for pipeline execution."""
    pipeline_id: str = Field(description="Pipeline identifier")
    pipeline_name: str = Field(description="Pipeline name")
    global_constraints: List[str] = Field(default_factory=list, description="Global constraints")
    execution_phase: str = Field(description="Current execution phase")
    shared_variables: Dict[str, str] = Field(default_factory=dict, description="Shared string variables")
    shared_numbers: Dict[str, float] = Field(default_factory=dict, description="Shared numeric variables")
    shared_flags: Dict[str, bool] = Field(default_factory=dict, description="Shared boolean flags")
    shared_lists: Dict[str, List[str]] = Field(default_factory=dict, description="Shared list variables")


class Pipeline:
    """Main pipeline orchestrator with fluent interface."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[PipelineStep] = []
        self.global_constraints: List[str] = []
        self.shared_context = SharedPipelineContext(
            pipeline_id=str(uuid.uuid4()),
            pipeline_name=name,
            global_constraints=[],
            execution_phase="init"
        )
        self.created_at = datetime.now()
        self.pipeline_id = self.shared_context.pipeline_id
        
    def step(self, name: str, goal: str, **kwargs) -> 'Pipeline':
        """Add step with fluent interface."""
        step = PipelineStep(name=name, goal=goal, **kwargs)
        self.steps.append(step)
        return self
    
    def add_constraint(self, constraint: str) -> 'Pipeline':
        """Add global constraint with fluent interface."""
        self.global_constraints.append(constraint)
        self.shared_context.global_constraints.append(constraint)
        return self
    
    def set_shared_variable(self, key: str, value: str) -> 'Pipeline':
        """Set shared string variable with fluent interface."""
        self.shared_context.shared_variables[key] = value
        return self
    
    def set_shared_number(self, key: str, value: float) -> 'Pipeline':
        """Set shared numeric variable with fluent interface."""
        self.shared_context.shared_numbers[key] = value
        return self
    
    def set_shared_flag(self, key: str, value: bool) -> 'Pipeline':
        """Set shared boolean flag with fluent interface."""
        self.shared_context.shared_flags[key] = value
        return self
    
    def set_shared_list(self, key: str, value: List[str]) -> 'Pipeline':
        """Set shared list variable with fluent interface."""
        self.shared_context.shared_lists[key] = value
        return self
    
    def set_execution_phase(self, phase: str) -> 'Pipeline':
        """Set execution phase with fluent interface."""
        self.shared_context.execution_phase = phase
        return self
    
    def get_step_by_name(self, name: str) -> Optional[PipelineStep]:
        """Get step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def get_steps_by_status(self, status: StepStatus) -> List[PipelineStep]:
        """Get steps by status."""
        return [step for step in self.steps if step.status == status]
    
    def validate(self) -> List[str]:
        """Validate pipeline definition."""
        errors = []
        step_names = [step.name for step in self.steps]
        
        # Check for duplicate step names
        if len(step_names) != len(set(step_names)):
            errors.append("Duplicate step names found")
        
        # Check dependencies exist
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_names:
                    errors.append(f"Step '{step.name}' depends on unknown step '{dep}'")
        
        # Check for empty pipeline
        if not self.steps:
            errors.append("Pipeline must contain at least one step")
        
        return errors
    
    def get_ready_steps(self, completed_steps: List[str]) -> List[PipelineStep]:
        """Get steps that are ready to execute based on completed dependencies."""
        ready_steps = []
        
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
                
            # Check if all dependencies are completed
            dependencies_met = all(dep in completed_steps for dep in step.depends_on)
            
            if dependencies_met:
                ready_steps.append(step)
        
        return ready_steps
    
    def get_step_dependencies(self, step_name: str) -> List[str]:
        """Get dependencies for a specific step."""
        step = self.get_step_by_name(step_name)
        return step.depends_on if step else []
    
    def get_step_dependents(self, step_name: str) -> List[str]:
        """Get steps that depend on the given step."""
        dependents = []
        for step in self.steps:
            if step_name in step.depends_on:
                dependents.append(step.name)
        return dependents
    
    def get_pipeline_summary(self) -> PipelineSummary:
        """Get summary of pipeline state."""
        status_counts = {}
        for status in StepStatus:
            status_counts[status.value] = len(self.get_steps_by_status(status))
        
        return PipelineSummary(
            pipeline_id=self.pipeline_id,
            name=self.name,
            description=self.description,
            total_steps=len(self.steps),
            status_counts=status_counts,
            created_at=self.created_at.isoformat(),
            global_constraints=self.global_constraints,
            shared_context_keys=(
                list(self.shared_context.shared_variables.keys()) +
                list(self.shared_context.shared_numbers.keys()) +
                list(self.shared_context.shared_flags.keys()) +
                list(self.shared_context.shared_lists.keys())
            )
        )


class PipelineExecutionSummary(BaseModel):
    """Comprehensive execution summary."""
    pipeline_name: str = Field(description="Pipeline name")
    pipeline_id: str = Field(description="Pipeline identifier")
    success: bool = Field(description="Overall success status")
    execution_time: float = Field(description="Total execution time")
    start_time: str = Field(description="Start timestamp")
    end_time: str = Field(description="End timestamp")
    total_cost: float = Field(description="Total execution cost")
    steps_completed: int = Field(description="Number of completed steps")
    steps_failed: int = Field(description="Number of failed steps")
    steps_skipped: int = Field(description="Number of skipped steps")
    total_steps: int = Field(description="Total number of steps")
    success_rate: float = Field(description="Success rate percentage")
    retry_count: int = Field(description="Total retry count")
    failed_steps: List[str] = Field(default_factory=list, description="List of failed steps")
    learned_facts_count: int = Field(description="Number of learned facts")
    saved_memories_count: int = Field(description="Number of saved memories")
    step_outputs_count: int = Field(description="Number of step outputs")


@dataclass
class PipelineResult:
    """Enhanced pipeline result with structured outputs and cost tracking."""
    pipeline_name: str
    pipeline_id: str
    success: bool
    execution_time: float
    start_time: datetime
    end_time: datetime
    
    # Cost tracking (existing TAgent functionality)
    total_cost: float = 0.0
    cost_per_step: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, TokenUsage] = field(default_factory=dict)
    
    # Structured outputs per step
    step_outputs: Dict[str, BaseModel] = field(default_factory=dict)
    step_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Optional final aggregated output
    final_output: Optional[BaseModel] = None
    
    # Learning and memory artifacts
    learned_facts: Dict[str, Any] = field(default_factory=dict)
    saved_memories: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    failed_steps: List[str] = field(default_factory=list)
    error_details: Dict[str, str] = field(default_factory=dict)
    
    # Execution statistics
    steps_completed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    total_steps: int = 0
    retry_count: int = 0
    
    def get_step_output(self, step_name: str, expected_type: Type[BaseModel]) -> Optional[BaseModel]:
        """Get typed output from a specific step."""
        output = self.step_outputs.get(step_name)
        if output and isinstance(output, expected_type):
            return output
        return None
    
    def get_final_output_or_last_step(self) -> Optional[BaseModel]:
        """Get final output or output from last successful step."""
        if self.final_output:
            return self.final_output
        
        # Get last successful step output
        if self.step_outputs:
            # Find the last completed step chronologically
            last_step = max(self.step_outputs.keys(), key=lambda k: self.step_metadata.get(k, {}).get('end_time', datetime.min))
            return self.step_outputs[last_step]
        
        return None
    
    def add_step_result(self, step_name: str, result: Any, metadata: Dict[str, Any] = None):
        """Add result for a step."""
        if isinstance(result, BaseModel):
            self.step_outputs[step_name] = result
        elif isinstance(result, dict):
            # Convert dict to generic BaseModel if needed
            class GenericStepOutput(BaseModel):
                data: Dict[str, Any]
            self.step_outputs[step_name] = GenericStepOutput(data=result)
        
        self.step_metadata[step_name] = metadata or {}
    
    def add_step_error(self, step_name: str, error_message: str):
        """Add error for a step."""
        self.failed_steps.append(step_name)
        self.error_details[step_name] = error_message
    
    def add_token_usage(self, step_name: str, usage: TokenUsage):
        """Add token usage for a step."""
        self.token_usage[step_name] = usage
        self.total_cost += usage.cost
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.steps_completed / self.total_steps) * 100
    
    def get_execution_summary(self) -> PipelineExecutionSummary:
        """Get comprehensive execution summary."""
        return PipelineExecutionSummary(
            pipeline_name=self.pipeline_name,
            pipeline_id=self.pipeline_id,
            success=self.success,
            execution_time=self.execution_time,
            start_time=self.start_time.isoformat(),
            end_time=self.end_time.isoformat(),
            total_cost=self.total_cost,
            steps_completed=self.steps_completed,
            steps_failed=self.steps_failed,
            steps_skipped=self.steps_skipped,
            total_steps=self.total_steps,
            success_rate=self.get_success_rate(),
            retry_count=self.retry_count,
            failed_steps=self.failed_steps,
            learned_facts_count=len(self.learned_facts),
            saved_memories_count=len(self.saved_memories),
            step_outputs_count=len(self.step_outputs)
        )


# Default output formats for pipeline steps
class DefaultStepOutput(BaseModel):
    """Default output format for pipeline steps."""
    result: Any = Field(description="The main result of the step")
    summary: str = Field(description="Summary of what was accomplished")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    success: bool = Field(default=True, description="Whether the step succeeded")


# Typed models for better type safety

class ExecutionMetadata(BaseModel):
    """Metadata for step execution."""
    current_phase: str = Field(description="Current pipeline phase")
    retry_count: int = Field(default=0, description="Current retry count")
    execution_time: str = Field(description="Execution timestamp")
    timeout: Optional[int] = Field(default=None, description="Step timeout in seconds")
    tools_filter: Optional[List[str]] = Field(default=None, description="Filtered tools for this step")


class StepExecutionSummary(BaseModel):
    """Summary of step execution."""
    step_name: str = Field(description="Name of the step")
    status: str = Field(description="Execution status")
    execution_time: Optional[float] = Field(default=None, description="Execution duration in seconds")
    retry_count: int = Field(default=0, description="Number of retries")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    result_available: bool = Field(default=False, description="Whether result is available")


class PipelineExecutionProgress(BaseModel):
    """Progress tracking for pipeline execution."""
    total_steps: int = Field(description="Total number of steps")
    completed_steps: int = Field(description="Number of completed steps")
    failed_steps: int = Field(description="Number of failed steps")
    running_steps: int = Field(description="Number of running steps")
    pending_steps: int = Field(description="Number of pending steps")
    progress_percentage: float = Field(description="Progress as percentage")
    current_phase: str = Field(description="Current pipeline phase")
    ready_steps: List[str] = Field(default_factory=list, description="Steps ready for execution")
    current_step: Optional[str] = Field(default=None, description="Currently executing step")


class PipelineMemoryState(BaseModel):
    """Complete memory state for a pipeline."""
    pipeline_id: str = Field(description="Unique pipeline identifier")
    step_results_count: int = Field(description="Number of step results stored")
    shared_data_keys: List[str] = Field(default_factory=list, description="Keys in shared data")
    execution_history_count: int = Field(description="Number of execution history events")
    current_step: Optional[str] = Field(default=None, description="Currently executing step")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Last update timestamp")
    completed_steps: List[str] = Field(default_factory=list, description="List of completed steps")


class StepContext(BaseModel):
    """Context for step execution."""
    step_name: str = Field(description="Name of the step")
    pipeline_id: str = Field(description="Pipeline identifier")
    shared_data_keys: List[str] = Field(default_factory=list, description="Available shared data keys")
    execution_history_count: int = Field(description="Number of execution history events")
    current_step: Optional[str] = Field(default=None, description="Currently executing step")
    timestamp: str = Field(description="Context creation timestamp")


class PipelineStepContext(BaseModel):
    """Context passed to pipeline steps during execution."""
    step_name: str = Field(description="Name of the current step")
    pipeline_name: str = Field(description="Name of the pipeline")
    pipeline_id: str = Field(description="Unique pipeline identifier")
    step_dependencies: List[str] = Field(default_factory=list, description="Names of dependency steps")
    dependency_results: Dict[str, BaseModel] = Field(default_factory=dict, description="Results from dependency steps")
    shared_context: SharedPipelineContext = Field(description="Shared pipeline context")
    execution_metadata: ExecutionMetadata = Field(description="Execution metadata")
    retry_count: int = Field(default=0, description="Current retry count for this step")
    max_retries: int = Field(default=3, description="Maximum retries allowed")


class PipelineExecutionSummary(BaseModel):
    """Comprehensive execution summary."""
    pipeline_name: str = Field(description="Pipeline name")
    pipeline_id: str = Field(description="Pipeline identifier")
    success: bool = Field(description="Overall success status")
    execution_time: float = Field(description="Total execution time")
    start_time: str = Field(description="Start timestamp")
    end_time: str = Field(description="End timestamp")
    total_cost: float = Field(description="Total execution cost")
    steps_completed: int = Field(description="Number of completed steps")
    steps_failed: int = Field(description="Number of failed steps")
    steps_skipped: int = Field(description="Number of skipped steps")
    total_steps: int = Field(description="Total number of steps")
    success_rate: float = Field(description="Success rate percentage")
    retry_count: int = Field(description="Total retry count")
    failed_steps: List[str] = Field(default_factory=list, description="List of failed steps")
    learned_facts_count: int = Field(description="Number of learned facts")
    saved_memories_count: int = Field(description="Number of saved memories")
    step_outputs_count: int = Field(description="Number of step outputs")


class PersistenceManagerSummary(BaseModel):
    """Summary of persistence manager state."""
    backend_type: str = Field(description="Type of storage backend")
    base_path: str = Field(description="Base storage path")
    backup_enabled: bool = Field(description="Whether backups are enabled")
    retention_days: int = Field(description="Data retention period in days")
    auto_cleanup: bool = Field(description="Whether auto cleanup is enabled")
    active_pipelines: List[str] = Field(default_factory=list, description="List of active pipeline IDs")
    shared_memory_keys: List[str] = Field(default_factory=list, description="Keys in shared memory")


class Checkpoint(BaseModel):
    """Represents a snapshot of pipeline state for recovery."""
    checkpoint_id: str = Field(description="Unique checkpoint identifier")
    pipeline_id: str = Field(description="Identifier of the pipeline being checkpointed")
    checkpoint_type: str = Field(description="Type or reason for the checkpoint (e.g., 'pre_step', 'manual')")
    state: PipelineMemoryState = Field(description="The pipeline memory state at the time of checkpoint")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of checkpoint creation")


class ExecutionHistoryEvent(BaseModel):
    """Represents a single event in the pipeline's execution history."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique event identifier")
    execution_id: str = Field(description="Identifier for the specific pipeline run")
    pipeline_id: str = Field(description="Identifier of the pipeline")
    event_type: str = Field(description="Type of event (e.g., 'execution_start', 'step_completed')")
    step_name: Optional[str] = Field(default=None, description="Name of the step involved, if applicable")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the event")
    details: Dict[str, Any] = Field(default_factory=dict, description="Event-specific details")


class AuditLog(BaseModel):
    """Represents a log entry for auditing purposes."""
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique log identifier")
    pipeline_id: str = Field(description="Identifier of the pipeline related to the event")
    event_name: str = Field(description="Name of the event being logged (e.g., 'state_saved', 'state_loaded')")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the event")
    details: Dict[str, Any] = Field(default_factory=dict, description="Log-specific details")


class PipelineProgress(BaseModel):
    """Real-time progress of a pipeline execution."""
    pipeline_id: str = Field(description="Unique pipeline identifier")
    total_steps: int = Field(description="Total number of steps in the pipeline")
    completed_steps: int = Field(default=0, description="Number of completed steps")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time of the pipeline execution")
    step_status: Dict[str, str] = Field(default_factory=dict, description="Status of each step")

    def update_step(self, step_name: str, status: str, result: Any = None):
        self.step_status[step_name] = status
        if status == StepStatus.COMPLETED.value:
            self.completed_steps += 1


class PipelineMetrics(BaseModel):
    """Performance metrics for a pipeline execution."""
    pipeline_id: str = Field(description="Unique pipeline identifier")
    total_duration: float = Field(description="Total execution duration in seconds")
    avg_step_duration: float = Field(description="Average duration of a single step")
    max_step_duration: float = Field(description="Maximum duration of any single step")
    min_step_duration: float = Field(description="Minimum duration of any single step")
    error_count: int = Field(description="Total number of errors")
    steps_completed: int = Field(description="Number of completed steps")
    resource_usage_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of resource usage")


class FailureReport(BaseModel):
    """Detailed report of a pipeline failure."""
    pipeline_id: str = Field(description="Identifier of the failed pipeline")
    error_type: str = Field(description="Type of the error that caused the failure")
    error_message: str = Field(description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the failure")
    context: Dict[str, Any] = Field(description="Pipeline context at the time of failure")
    analysis: Dict[str, Any] = Field(description="Automated analysis of the failure")
    recommendations: List[str] = Field(description="Recommendations to prevent future failures")


class PipelineTemplate(BaseModel):
    """A reusable template for creating pipelines."""
    name: str = Field(description="Unique name for the template")
    description: str = Field(description="Description of the template's purpose")
    parameters: Dict[str, Any] = Field(description="Dictionary of parameters the template accepts")
    pipeline_factory: Any = Field(description="A callable that returns a Pipeline instance")

    def create_pipeline(self, parameters: Dict[str, Any]) -> "Pipeline":
        """Create a new pipeline instance from the template."""
        # Basic validation
        for name, spec in self.parameters.items():
            if spec.get("required") and name not in parameters:
                raise ValueError(f"Missing required parameter: {name}")
        
        # Fill in defaults
        filled_params = self.parameters.copy()
        for name, spec in filled_params.items():
            if "default" in spec:
                parameters.setdefault(name, spec["default"])

        return self.pipeline_factory(parameters)



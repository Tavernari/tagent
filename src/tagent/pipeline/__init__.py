"""
TAgent Pipeline System

A sophisticated workflow orchestration feature that enables complex multi-step AI agent 
execution with memory persistence and inter-pipeline communication.
"""

from .models import (
    PipelineStep, 
    Pipeline, 
    PipelineResult, 
    ExecutionMode, 
    DependencyType, 
    StepStatus,
    DefaultStepOutput,
    PipelineStepContext,
    ExecutionMetadata,
    StepExecutionSummary,
    PipelineExecutionProgress,
    PipelineMemoryState,
    PipelineSummary,
    StepContext,
    SharedPipelineContext,
    PipelineExecutionSummary,
    PersistenceManagerSummary,
    Checkpoint,
    ExecutionHistoryEvent,
    AuditLog
)
from .scheduler import PipelineScheduler, SchedulingStrategy, SchedulingSummary
from .state import PipelineStateMachine, PipelineMemory, PipelinePhase
from .persistence import (
    PipelineMemoryManager, 
    PersistenceConfig, 
    StorageBackendType,
    FileStorageBackend,
    SQLiteStorageBackend,
    MemoryStorageBackend,
    CheckpointManager,
    HistoryManager,
    AuditManager,
    CheckpointNotFoundError,
)
from .executor import PipelineExecutor, PipelineExecutorConfig, PipelineValidationError
from .step_executor import StepExecutor, RetryConfig, TimeoutConfig
from .communication import (
    PipelineCommunicator, PipelineMessage, PipelineEvent, PipelineInfo,
    EventSubscriber, SharedMemorySpace, MessageType, EventType, MessagePriority
)
from .conditions import (
    ConditionEvaluator,
    ConditionDSL,
    AnyCondition,
    ValueReference,
    EqualsCondition,
    NotEqualsCondition,
    ContainsCondition,
    GreaterThanCondition,
    LessThanCondition,
    ExistsCondition,
    NotExistsCondition,
    AndCondition,
    OrCondition,
)

__all__ = [
    # Models
    "PipelineStep",
    "Pipeline", 
    "PipelineResult",
    "ExecutionMode",
    "DependencyType",
    "StepStatus",
    "DefaultStepOutput",
    "PipelineStepContext",
    "ExecutionMetadata",
    "StepExecutionSummary",
    "PipelineExecutionProgress",
    "PipelineMemoryState",
    "PipelineSummary",
    "StepContext",
    "SharedPipelineContext",
    "PipelineExecutionSummary",
    "PersistenceManagerSummary",
    "Checkpoint",
    "ExecutionHistoryEvent",
    "AuditLog",
    
    # Scheduler
    "PipelineScheduler",
    "SchedulingStrategy",
    "SchedulingSummary",
    
    # State Management
    "PipelineStateMachine",
    "PipelineMemory",
    "PipelinePhase",
    
    # Persistence
    "PipelineMemoryManager",
    "PersistenceConfig", 
    "StorageBackendType",
    "FileStorageBackend",
    "SQLiteStorageBackend",
    "MemoryStorageBackend",
    "CheckpointManager",
    "HistoryManager",
    "AuditManager",
    "CheckpointNotFoundError",
    
    # Execution
    "PipelineExecutor",
    "PipelineExecutorConfig", 
    "PipelineValidationError",
    "StepExecutor",
    "RetryConfig",
    "TimeoutConfig",
    
    # Communication
    "PipelineCommunicator",
    "PipelineMessage",
    "PipelineEvent", 
    "PipelineInfo",
    "EventSubscriber",
    "SharedMemorySpace",
    "MessageType",
    "EventType", 
    "MessagePriority",

    # Conditions
    "ConditionEvaluator",
    "ConditionDSL",
    "AnyCondition",
    "ValueReference",
    "EqualsCondition",
    "NotEqualsCondition",
    "ContainsCondition",
    "GreaterThanCondition",
    "LessThanCondition",
    "ExistsCondition",
    "NotExistsCondition",
    "AndCondition",
    "OrCondition",
]
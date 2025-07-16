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
    PipelineStepContext
)
from .scheduler import PipelineScheduler, SchedulingStrategy
from .state import PipelineStateMachine, PipelineMemory, PipelinePhase
from .persistence import (
    PipelineMemoryManager, 
    PersistenceConfig, 
    StorageBackendType,
    FileStorageBackend,
    SQLiteStorageBackend,
    MemoryStorageBackend
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
    
    # Scheduler
    "PipelineScheduler",
    "SchedulingStrategy",
    
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
    "MemoryStorageBackend"
]
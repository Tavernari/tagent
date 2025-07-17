"""
TAgent Pipeline System

A sophisticated workflow orchestration feature that enables complex multi-step AI agent 
execution with memory persistence and inter-pipeline communication.
"""

# --- Feature Availability ---
_PIPELINE_CORE_AVAILABLE = False
_PIPELINE_MONITORING_AVAILABLE = False
_PIPELINE_PERSISTENCE_AVAILABLE = False
_PIPELINE_TEMPLATES_AVAILABLE = False
_PIPELINE_API_AVAILABLE = False

def _create_feature_checker(feature_name: str, extra: str):
    def require_feature():
        raise ImportError(
            f"Pipeline {feature_name} features are not available. "
            f"Install with: pip install tagent[{extra}]"
        )
    return require_feature

# --- Core Imports ---
try:
    from .models import (
        PipelineStep, Pipeline, PipelineResult, ExecutionMode, DependencyType, 
        StepStatus, DefaultStepOutput, PipelineStepContext, ExecutionMetadata,
        StepExecutionSummary, PipelineExecutionProgress, PipelineMemoryState,
        PipelineSummary, StepContext, SharedPipelineContext, PipelineExecutionSummary
    )
    from .scheduler import PipelineScheduler, SchedulingStrategy, SchedulingSummary
    from .state import PipelineStateMachine, PipelineMemory, PipelinePhase
    from .executor import PipelineExecutor, PipelineExecutorConfig
    from .step_executor import StepExecutor, RetryConfig, TimeoutConfig
    from .communication import (
        PipelineCommunicator, PipelineMessage, PipelineEvent, PipelineInfo,
        EventSubscriber, SharedMemorySpace, MessageType, EventType, MessagePriority
    )
    from .conditions import (
        ConditionEvaluator, ConditionDSL, AnyCondition, ValueReference,
        EqualsCondition, NotEqualsCondition, ContainsCondition, GreaterThanCondition,
        LessThanCondition, ExistsCondition, NotExistsCondition, AndCondition, OrCondition
    )
    _PIPELINE_CORE_AVAILABLE = True
except ImportError:
    require_core = _create_feature_checker("core", "pipeline")
    class Pipeline:
        def __init__(self, *args, **kwargs): require_core()
    class PipelineStep:
        def __init__(self, *args, **kwargs): require_core()
    class PipelineResult:
        def __init__(self, *args, **kwargs): require_core()
    class ExecutionMode: SERIAL="serial"; CONCURRENT="concurrent"

# --- Persistence Imports ---
try:
    from .persistence import (
        PipelineMemoryManager, PersistenceConfig, StorageBackendType,
        FileStorageBackend, SQLiteStorageBackend, MemoryStorageBackend,
        CheckpointManager, HistoryManager, AuditManager, CheckpointNotFoundError,
    )
    from .models import PersistenceManagerSummary, Checkpoint, ExecutionHistoryEvent, AuditLog
    _PIPELINE_PERSISTENCE_AVAILABLE = True
except ImportError:
    require_persistence = _create_feature_checker("persistence", "persistence")
    class PipelineMemoryManager:
        def __init__(self, *args, **kwargs): require_persistence()

# --- Monitoring Imports ---
try:
    from .monitoring import (
        PipelineMonitor, MetricsCollector, FailureAnalyzer, AlertManager
    )
    from .models import PipelineProgress, PipelineMetrics, FailureReport
    _PIPELINE_MONITORING_AVAILABLE = True
except ImportError:
    require_monitoring = _create_feature_checker("monitoring", "monitoring")
    class PipelineMonitor:
        def __init__(self, *args, **kwargs): require_monitoring()

# --- Template Imports ---
try:
    from .templates import (
        PipelineTemplateRegistry, TemplateNotFoundError,
        COMPANY_RESEARCH_TEMPLATE, ECOMMERCE_ANALYSIS_TEMPLATE
    )
    from .models import PipelineTemplate
    _PIPELINE_TEMPLATES_AVAILABLE = True
except ImportError:
    require_templates = _create_feature_checker("templates", "pipeline")
    class PipelineTemplateRegistry:
        def __init__(self, *args, **kwargs): require_templates()

# --- API Imports ---
try:
    from .api import (
        PipelineBuilder, PipelineOptimizer, PipelineSerializer, PipelineValidationError
    )
    _PIPELINE_API_AVAILABLE = True
except ImportError:
    require_api = _create_feature_checker("api", "pipeline")
    class PipelineBuilder:
        def __init__(self, *args, **kwargs): require_api()


def check_feature_availability() -> dict:
    """Check which pipeline features are available."""
    return {
        'core': _PIPELINE_CORE_AVAILABLE,
        'monitoring': _PIPELINE_MONITORING_AVAILABLE,
        'persistence': _PIPELINE_PERSISTENCE_AVAILABLE,
        'templates': _PIPELINE_TEMPLATES_AVAILABLE,
        'api': _PIPELINE_API_AVAILABLE,
    }

__all__ = [
    # Feature Availability
    "check_feature_availability",

    # Core
    "PipelineStep", "Pipeline", "PipelineResult", "ExecutionMode", "DependencyType", 
    "StepStatus", "DefaultStepOutput", "PipelineStepContext", "ExecutionMetadata",
    "StepExecutionSummary", "PipelineExecutionProgress", "PipelineMemoryState",
    "PipelineSummary", "StepContext", "SharedPipelineContext", "PipelineExecutionSummary",
    "PipelineScheduler", "SchedulingStrategy", "SchedulingSummary",
    "PipelineStateMachine", "PipelineMemory", "PipelinePhase",
    "PipelineExecutor", "PipelineExecutorConfig",
    "StepExecutor", "RetryConfig", "TimeoutConfig",
    "PipelineCommunicator", "PipelineMessage", "PipelineEvent", "PipelineInfo",
    "EventSubscriber", "SharedMemorySpace", "MessageType", "EventType", "MessagePriority",
    "ConditionEvaluator", "ConditionDSL", "AnyCondition", "ValueReference",
    "EqualsCondition", "NotEqualsCondition", "ContainsCondition", "GreaterThanCondition",
    "LessThanCondition", "ExistsCondition", "NotExistsCondition", "AndCondition", "OrCondition",

    # Persistence
    "PipelineMemoryManager", "PersistenceConfig", "StorageBackendType",
    "FileStorageBackend", "SQLiteStorageBackend", "MemoryStorageBackend",
    "CheckpointManager", "HistoryManager", "AuditManager", "CheckpointNotFoundError",
    "PersistenceManagerSummary", "Checkpoint", "ExecutionHistoryEvent", "AuditLog",

    # Monitoring
    "PipelineMonitor", "MetricsCollector", "FailureAnalyzer", "AlertManager",
    "PipelineProgress", "PipelineMetrics", "FailureReport",

    # Templates
    "PipelineTemplateRegistry", "TemplateNotFoundError", "COMPANY_RESEARCH_TEMPLATE", 
    "ECOMMERCE_ANALYSIS_TEMPLATE", "PipelineTemplate",

    # API
    "PipelineBuilder", "PipelineOptimizer", "PipelineSerializer", "PipelineValidationError",
]

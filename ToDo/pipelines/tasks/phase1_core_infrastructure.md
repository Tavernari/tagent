# Phase 1: Core Infrastructure Tasks

## Task 1.1: Pipeline Models (`src/tagent/pipeline/models.py`)
**Priority: HIGH** | **Estimated Time: 3-4 days**

### Objective
Create the foundational data models for pipeline definition and execution.

### Requirements
- [ ] Create `PipelineStep` dataclass with all required fields
- [ ] **NEW**: Add `output_schema` field for Pydantic model definitions
- [ ] Create `Pipeline` class with fluent builder interface
- [ ] Create enhanced `PipelineResult` class with structured outputs
- [ ] **NEW**: Add `step_outputs` dict for structured results per step
- [ ] **NEW**: Add `learned_facts` and `saved_memories` fields
- [ ] **NEW**: Add cost tracking fields (total_cost, cost_per_step, token_usage)
- [ ] Create `ExecutionMode` and `DependencyType` enums
- [ ] Add validation logic for pipeline definitions
- [ ] Add serialization/deserialization methods

### Implementation Details

#### PipelineStep Model
```python
@dataclass
class PipelineStep:
    """Individual step in a pipeline."""
    name: str                                    # Unique step name (used for dependencies)
    goal: str                                    # Step objective
    constraints: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # Step names, not indices
    execution_mode: ExecutionMode = ExecutionMode.SERIAL
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    tools_filter: Optional[List[str]] = None
    output_schema: Optional[Type[BaseModel]] = None  # NEW: User-defined output schema
    
    def __post_init__(self):
        """Validate step configuration."""
        if not self.name:
            raise ValueError("Step name cannot be empty")
        if not self.goal:
            raise ValueError("Step goal cannot be empty")
    
    def validate_output(self, result: Any) -> BaseModel:
        """Validate step output against schema."""
        if self.output_schema:
            if isinstance(result, dict):
                return self.output_schema(**result)
            elif isinstance(result, self.output_schema):
                return result
            else:
                raise ValueError(f"Step '{self.name}' output does not match schema {self.output_schema}")
        return result
```

#### Pipeline Model
```python
class Pipeline:
    """Main pipeline orchestrator with fluent interface."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[PipelineStep] = []
        self.global_constraints: List[str] = []
        self.shared_context: Dict[str, Any] = {}
        
    def step(self, name: str, goal: str, **kwargs) -> 'Pipeline':
        """Add step with fluent interface."""
        step = PipelineStep(name=name, goal=goal, **kwargs)
        self.steps.append(step)
        return self
    
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
        
        return errors

#### Enhanced PipelineResult Model
```python
@dataclass
class PipelineResult:
    """Enhanced pipeline result with structured outputs and cost tracking."""
    pipeline_name: str
    success: bool
    execution_time: float
    
    # Cost tracking (existing TAgent functionality)
    total_cost: float
    cost_per_step: Dict[str, float]
    token_usage: Dict[str, TokenUsage]
    
    # NEW: Structured outputs per step
    step_outputs: Dict[str, BaseModel]  # Step name -> Validated output
    step_metadata: Dict[str, Dict[str, Any]]  # Step name -> Metadata
    
    # NEW: Optional final aggregated output
    final_output: Optional[BaseModel] = None
    
    # NEW: Learning and memory artifacts
    learned_facts: Dict[str, Any] = field(default_factory=dict)
    saved_memories: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    failed_steps: List[str] = field(default_factory=list)
    error_details: Dict[str, str] = field(default_factory=dict)
    
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
            last_step = max(self.step_outputs.keys())
            return self.step_outputs[last_step]
        
        return None
```
```

### Acceptance Criteria
- [ ] All models have comprehensive type hints
- [ ] Validation prevents invalid pipeline configurations
- [ ] Serialization supports JSON and pickle formats
- [ ] Fluent interface works as expected
- [ ] Error messages are clear and actionable

---

## Task 1.2: Pipeline Scheduler (`src/tagent/pipeline/scheduler.py`)
**Priority: HIGH** | **Estimated Time: 4-5 days**

### Objective
Implement dependency resolution and execution scheduling for pipeline steps.

### Requirements
- [ ] Implement dependency graph builder using step names
- [ ] Create topological sort algorithm for execution order
- [ ] Add circular dependency detection
- [ ] Implement `get_ready_steps()` method
- [ ] Add step status tracking
- [ ] Create deadlock detection mechanism

### Implementation Details

#### Dependency Graph Builder
```python
class PipelineScheduler:
    """Manages pipeline execution order and dependencies."""
    
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.step_status: Dict[str, StepStatus] = {}
        self.dependency_graph = self._build_dependency_graph()
        self._validate_dependencies()
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph using step names."""
        graph = {}
        for step in self.pipeline.steps:
            graph[step.name] = step.depends_on.copy()
        return graph
    
    def _validate_dependencies(self):
        """Validate dependency graph for cycles."""
        if self._has_circular_dependencies():
            raise ValueError("Circular dependencies detected in pipeline")
```

#### Topological Sort Implementation
```python
def get_execution_order(self) -> List[str]:
    """Get topological order of step execution."""
    visited = set()
    temp_visited = set()
    result = []
    
    def dfs(step_name: str):
        if step_name in temp_visited:
            raise ValueError(f"Circular dependency detected involving step '{step_name}'")
        if step_name in visited:
            return
        
        temp_visited.add(step_name)
        for dependency in self.dependency_graph.get(step_name, []):
            dfs(dependency)
        
        temp_visited.remove(step_name)
        visited.add(step_name)
        result.append(step_name)
    
    for step_name in self.dependency_graph:
        if step_name not in visited:
            dfs(step_name)
    
    return result
```

### Acceptance Criteria
- [ ] Circular dependency detection works correctly
- [ ] Topological sort produces valid execution order
- [ ] Ready steps are identified correctly
- [ ] Status tracking is accurate
- [ ] Performance is acceptable for large pipelines

---

## Task 1.3: Pipeline State Management (`src/tagent/pipeline/state.py`)
**Priority: HIGH** | **Estimated Time: 5-6 days**

### Objective
Extend existing state machine to support pipeline execution with memory persistence.

### Requirements
- [ ] Extend `TaskBasedStateMachine` for pipeline context
- [ ] Add shared context management between steps
- [ ] Implement step progress tracking
- [ ] Add pipeline resumption logic
- [ ] Create state serialization for checkpointing

### Implementation Details

#### Pipeline Memory Management
```python
class PipelineMemory:
    """Enhanced memory management for pipeline execution."""
    
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.step_results: Dict[str, StepMemoryEntry] = {}
        self.shared_data: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.current_step: Optional[str] = None
        
    def save_step_result(self, step_name: str, result: Any):
        """Save step result with timestamp and metadata."""
        entry = StepMemoryEntry(
            step_name=step_name,
            pipeline_id=self.pipeline_id,
            data=result,
            timestamp=datetime.now(),
            execution_id=str(uuid.uuid4()),
            dependencies_used=self._get_used_dependencies(step_name),
            metadata={}
        )
        self.step_results[step_name] = entry
        self._update_execution_history(step_name, "completed", result)
    
    def get_step_result(self, step_name: str) -> Any:
        """Get result from specific step."""
        entry = self.step_results.get(step_name)
        return entry.data if entry else None
    
    def has_step_result(self, step_name: str) -> bool:
        """Check if step has completed with result."""
        return step_name in self.step_results
```

#### Extended State Machine
```python
class PipelineStateMachine(TaskBasedStateMachine):
    """Enhanced state machine for pipeline execution."""
    
    def __init__(self, pipeline: Pipeline, memory_manager: PipelineMemoryManager):
        super().__init__(pipeline.description, [])
        self.pipeline = pipeline
        self.memory_manager = memory_manager
        self.pipeline_memory = memory_manager.get_pipeline_memory(pipeline.name)
        self.scheduler = PipelineScheduler(pipeline)
        
    def get_ready_steps(self) -> List[PipelineStep]:
        """Get steps ready for execution based on dependencies."""
        ready_steps = []
        for step in self.pipeline.steps:
            if self._can_execute_step(step):
                ready_steps.append(step)
        return ready_steps
    
    def _can_execute_step(self, step: PipelineStep) -> bool:
        """Check if step dependencies are satisfied."""
        # Check if step already completed
        if self.pipeline_memory.has_step_result(step.name):
            return False
        
        # Check all dependencies are satisfied
        for dependency in step.depends_on:
            if not self.pipeline_memory.has_step_result(dependency):
                return False
        
        return True
```

### Acceptance Criteria
- [ ] Memory persists across pipeline restarts
- [ ] Step dependencies are correctly tracked
- [ ] State can be serialized and restored
- [ ] Progress tracking is accurate
- [ ] Integration with existing state machine works

---

## Task 1.4: Memory Persistence Backend (`src/tagent/pipeline/persistence.py`)
**Priority: HIGH** | **Estimated Time: 4-5 days**

### Objective
Implement robust memory persistence for pipeline execution with multiple storage backends.

### Requirements
- [ ] Implement file-based storage (JSON/pickle)
- [ ] Add database storage option (SQLite)
- [ ] Create Redis storage for distributed execution
- [ ] Add automatic backup and recovery
- [ ] Implement data retention policies

### Implementation Details

#### Storage Backend Interface
```python
class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def save(self, pipeline_id: str, data: Dict[str, Any]):
        """Save pipeline memory data."""
        pass
    
    @abstractmethod
    async def load(self, pipeline_id: str) -> Dict[str, Any]:
        """Load pipeline memory data."""
        pass
    
    @abstractmethod
    async def delete(self, pipeline_id: str):
        """Delete pipeline memory data."""
        pass
    
    @abstractmethod
    async def list_pipelines(self) -> List[str]:
        """List all stored pipeline IDs."""
        pass
```

#### File-based Storage
```python
class FileStorageBackend(StorageBackend):
    """File-based storage backend."""
    
    def __init__(self, base_path: str = "./pipeline_memory"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def save(self, pipeline_id: str, data: Dict[str, Any]):
        """Save data to JSON file."""
        file_path = self.base_path / f"{pipeline_id}.json"
        
        # Create backup of existing file
        if file_path.exists():
            backup_path = self.base_path / f"{pipeline_id}.backup.json"
            shutil.copy2(file_path, backup_path)
        
        # Save new data
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
```

### Acceptance Criteria
- [ ] Data persists across system restarts
- [ ] Multiple storage backends work correctly
- [ ] Backup and recovery mechanisms function
- [ ] Performance is acceptable for large datasets
- [ ] Error handling is robust

---

## Integration Requirements

### Dependencies Between Tasks
- Task 1.1 (Models) must be completed before all other tasks
- Task 1.2 (Scheduler) depends on Task 1.1
- Task 1.3 (State Management) depends on Tasks 1.1 and 1.4
- Task 1.4 (Persistence) can be developed in parallel with Task 1.2

### Testing Requirements
- Unit tests for each component
- Integration tests between components
- Performance benchmarks
- Error handling tests
- Memory leak detection

### Documentation Requirements
- API documentation for all public methods
- Architecture decision records
- Usage examples
- Migration guides
- Troubleshooting guides

## Success Metrics
- All tasks completed with 100% test coverage
- Performance benchmarks meet requirements
- Memory usage stays within acceptable limits
- Error handling covers all edge cases
- Documentation is comprehensive and accurate
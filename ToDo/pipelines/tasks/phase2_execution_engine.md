# Phase 2: Execution Engine Tasks

## Task 2.1: Pipeline Executor (`src/tagent/pipeline/executor.py`)
**Priority: HIGH** | **Estimated Time: 6-7 days**

### Objective
Create the main pipeline execution engine with async support, memory persistence, and inter-pipeline communication.

### Requirements
- [ ] Create async `PipelineExecutor` class
- [ ] Implement main execution loop with dependency resolution
- [ ] Add concurrent step execution using `asyncio.gather`
- [ ] Integrate with existing `TaskBasedStateMachine`
- [ ] Add error handling and recovery strategies
- [ ] Implement resource management for LLM calls

### Implementation Details

#### Core Executor Class
```python
class PipelineExecutor:
    """Main pipeline execution engine with async support."""
    
    def __init__(self, pipeline: Pipeline, config: Dict[str, Any]):
        self.pipeline = pipeline
        self.config = config
        self.memory_manager = PipelineMemoryManager(config.get('storage_backend', 'file'))
        self.communicator = PipelineCommunicator()
        self.state_machine = PipelineStateMachine(pipeline, self.memory_manager)
        self.executor_pool = asyncio.Semaphore(config.get('max_concurrent_steps', 5))
        
    async def execute(self) -> PipelineResult:
        """Execute the complete pipeline."""
        print_retro_banner(f"PIPELINE: {self.pipeline.name}", "═", 70)
        
        try:
            # Initialize or restore pipeline state
            await self._initialize_pipeline_state()
            
            # Main execution loop
            while not self.state_machine.is_pipeline_complete():
                ready_steps = self.state_machine.get_ready_steps()
                
                if not ready_steps:
                    # Check for deadlock
                    if self._is_deadlocked():
                        raise PipelineDeadlockError("Pipeline execution deadlocked")
                    break
                
                # Execute ready steps
                await self._execute_step_batch(ready_steps)
                
                # Persist state after each batch
                await self._persist_pipeline_state()
            
            # Create final result
            result = await self._create_pipeline_result()
            
            # Cleanup resources
            await self._cleanup_resources()
            
            return result
            
        except Exception as e:
            # Handle errors with persistence for recovery
            await self._handle_pipeline_error(e)
            raise
```

#### Concurrent Step Execution
```python
async def _execute_step_batch(self, steps: List[PipelineStep]):
    """Execute a batch of steps with proper concurrency control."""
    serial_steps = []
    concurrent_steps = []
    
    # Separate steps by execution mode
    for step in steps:
        if step.execution_mode == ExecutionMode.CONCURRENT:
            concurrent_steps.append(step)
        else:
            serial_steps.append(step)
    
    # Execute serial steps first
    for step in serial_steps:
        await self._execute_single_step(step)
    
    # Execute concurrent steps in parallel
    if concurrent_steps:
        concurrent_tasks = [
            self._execute_single_step(step) 
            for step in concurrent_steps
        ]
        await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        # Handle any exceptions from concurrent execution
        await self._handle_concurrent_step_results(concurrent_tasks)

async def _execute_single_step(self, step: PipelineStep):
    """Execute individual step with memory persistence."""
    async with self.executor_pool:
        step_id = f"{self.pipeline.name}:{step.name}"
        
        try:
            # Load step context from memory
            context = await self._prepare_step_context(step)
            
            # Execute step using TAgent
            result = await self._execute_tagent_step(step, context)
            
            # Save result to persistent memory
            self.state_machine.save_step_result(step.name, result)
            
            # Broadcast completion event
            await self.communicator.broadcast_event(
                f"step_completed",
                {
                    'pipeline_id': self.pipeline.name,
                    'step_name': step.name,
                    'result': result,
                    'timestamp': datetime.now()
                }
            )
            
            print_retro_status("SUCCESS", f"Step '{step.name}' completed")
            
        except Exception as e:
            # Handle step failure with retry logic
            await self._handle_step_error(step, e)
            raise
```

### Acceptance Criteria
- [ ] Concurrent step execution works correctly
- [ ] Memory persistence is maintained throughout execution
- [ ] Error handling and recovery function properly
- [ ] Resource management prevents memory leaks
- [ ] Integration with existing TAgent systems works

---

## Task 2.2: Step Execution Wrapper (`src/tagent/pipeline/step_executor.py`)
**Priority: HIGH** | **Estimated Time: 4-5 days**

### Objective
Create a wrapper around existing `run_task_based_agent` to integrate with pipeline execution.

### Requirements
- [ ] Create wrapper for `run_task_based_agent`
- [ ] Add context injection for pipeline steps
- [ ] Implement result collection and formatting
- [ ] Add timeout handling per step
- [ ] Create retry logic with exponential backoff
- [ ] Add tool filtering per step

### Implementation Details

#### Step Execution Wrapper
```python
class StepExecutor:
    """Wrapper for executing individual pipeline steps."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout_manager = TimeoutManager()
        self.retry_manager = RetryManager()
        
    async def execute_step(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any],
        pipeline_memory: PipelineMemory
    ) -> Any:
        """Execute a single pipeline step with context injection."""
        
        # Prepare step execution context
        step_context = self._prepare_step_context(step, context, pipeline_memory)
        
        # Apply tool filtering if specified
        available_tools = self._filter_tools(step.tools_filter)
        
        # Execute with timeout and retry logic
        for attempt in range(step.max_retries + 1):
            try:
                # Execute step with timeout
                result = await self._execute_with_timeout(
                    step, step_context, available_tools
                )
                
                # Validate and format result
                formatted_result = self._format_step_result(step, result)
                
                return formatted_result
                
            except TimeoutError:
                if attempt < step.max_retries:
                    await self._handle_timeout_retry(step, attempt)
                else:
                    raise StepTimeoutError(f"Step '{step.name}' timed out after {step.max_retries} retries")
                    
            except Exception as e:
                if attempt < step.max_retries and self._is_retryable_error(e):
                    await self._handle_error_retry(step, e, attempt)
                else:
                    raise StepExecutionError(f"Step '{step.name}' failed: {str(e)}")

    def _prepare_step_context(
        self, 
        step: PipelineStep, 
        context: Dict[str, Any], 
        pipeline_memory: PipelineMemory
    ) -> Dict[str, Any]:
        """Prepare execution context with memory injection."""
        step_context = context.copy()
        
        # Inject dependency results
        for dependency in step.depends_on:
            dependency_result = pipeline_memory.get_step_result(dependency)
            if dependency_result:
                step_context[f"dependency_{dependency}"] = dependency_result
        
        # Add pipeline-wide shared data
        step_context.update(pipeline_memory.shared_data)
        
        # Add step-specific metadata
        step_context.update({
            'step_name': step.name,
            'pipeline_id': pipeline_memory.pipeline_id,
            'execution_attempt': 0,
            'constraints': step.constraints
        })
        
        return step_context
```

#### Timeout and Retry Management
```python
class TimeoutManager:
    """Manages step execution timeouts."""
    
    async def execute_with_timeout(self, coro, timeout: Optional[int]):
        """Execute coroutine with timeout."""
        if timeout is None:
            return await coro
        
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self):
        self.base_delay = 1.0
        self.max_delay = 60.0
        self.backoff_multiplier = 2.0
        
    async def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay)
    
    async def should_retry(self, error: Exception, attempt: int, max_retries: int) -> bool:
        """Determine if error should trigger retry."""
        if attempt >= max_retries:
            return False
        
        # Define retryable errors
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            APIRateLimitError,
            TemporaryServiceError
        )
        
        return isinstance(error, retryable_errors)
```

### Acceptance Criteria
- [ ] Step execution integrates seamlessly with existing TAgent
- [ ] Context injection works correctly
- [ ] Timeout handling prevents hanging steps
- [ ] Retry logic handles transient failures
- [ ] Tool filtering restricts available tools per step

---

## Task 2.3: Agent Interface Enhancement (`src/tagent/agent.py`)
**Priority: HIGH** | **Estimated Time: 2-3 days**

### Objective
Enhance the main `run_agent` function to support pipeline execution while maintaining backward compatibility.

### Requirements
- [ ] Modify `run_agent` to accept `Pipeline` objects
- [ ] Add `run_pipeline` function
- [ ] Implement graceful fallback when pipeline not available
- [ ] Ensure backward compatibility with existing calls
- [ ] Add configuration inheritance from base config

### Implementation Details

#### Enhanced Agent Interface
```python
def run_agent(
    goal_or_pipeline: Union[str, Pipeline],
    config: Optional['TAgentConfig'] = None,
    **kwargs
) -> Union[TaskBasedAgentResult, PipelineResult]:
    """
    Enhanced run_agent supporting both single goals and pipelines.
    
    Args:
        goal_or_pipeline: Either a goal string or Pipeline object
        config: Optional configuration object
        **kwargs: Additional parameters for backward compatibility
        
    Returns:
        TaskBasedAgentResult for single goals or PipelineResult for pipelines
    """
    # Check if pipeline functionality is available
    if isinstance(goal_or_pipeline, Pipeline):
        if not PIPELINE_AVAILABLE:
            raise ImportError(
                "Pipeline support requires additional dependencies. "
                "Install with: pip install tagent[pipeline]"
            )
        return asyncio.run(run_pipeline(goal_or_pipeline, config, **kwargs))
    else:
        # Existing single-goal execution
        return run_task_based_agent(goal_or_pipeline, config, **kwargs)

async def run_pipeline(
    pipeline: Pipeline,
    config: Optional['TAgentConfig'] = None,
    **kwargs
) -> PipelineResult:
    """
    Execute a pipeline with enhanced orchestration.
    
    Args:
        pipeline: Pipeline object to execute
        config: Optional configuration object
        **kwargs: Additional parameters
        
    Returns:
        PipelineResult with execution results and metadata
    """
    # Validate pipeline
    validation_errors = pipeline.validate()
    if validation_errors:
        raise PipelineValidationError(f"Pipeline validation failed: {validation_errors}")
    
    # Prepare configuration
    if config is None:
        config = TAgentConfig()
    
    # Override config with kwargs for backward compatibility
    config = _merge_config_with_kwargs(config, kwargs)
    
    # Create and execute pipeline
    executor = PipelineExecutor(pipeline, config.to_dict())
    result = await executor.execute()
    
    return result
```

#### Configuration Inheritance
```python
def _merge_config_with_kwargs(config: 'TAgentConfig', kwargs: Dict[str, Any]) -> 'TAgentConfig':
    """Merge configuration with kwargs for backward compatibility."""
    config_dict = config.to_dict()
    
    # Map kwargs to config parameters
    param_mapping = {
        'model': 'model',
        'api_key': 'api_key',
        'max_iterations': 'max_iterations',
        'tools': 'tools',
        'output_format': 'output_format',
        'verbose': 'verbose'
    }
    
    for kwarg, config_param in param_mapping.items():
        if kwarg in kwargs:
            config_dict[config_param] = kwargs[kwarg]
    
    return TAgentConfig.from_dict(config_dict)
```

#### Graceful Import Handling
```python
# At module level
try:
    from .pipeline import Pipeline, PipelineResult, PipelineExecutor
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    
    # Create dummy classes for type hints
    class Pipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError("Pipeline support not available")
    
    class PipelineResult:
        pass
```

### Acceptance Criteria
- [ ] Backward compatibility maintained for existing code
- [ ] Pipeline execution works when dependencies are available
- [ ] Graceful error messages when pipeline not available
- [ ] Configuration inheritance works correctly
- [ ] Type hints are preserved

---

## Task 2.4: Inter-Pipeline Communication (`src/tagent/pipeline/communication.py`)
**Priority: HIGH** | **Estimated Time: 5-6 days**

### Objective
Implement robust inter-pipeline communication system for data sharing and event broadcasting.

### Requirements
- [ ] Create message queue system for pipeline communication
- [ ] Implement event-driven communication patterns
- [ ] Add shared memory spaces for related pipelines
- [ ] Create pipeline registry and discovery
- [ ] Add communication monitoring and logging

### Implementation Details

#### Pipeline Communicator
```python
class PipelineCommunicator:
    """Handles communication between pipelines."""
    
    def __init__(self):
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.event_subscribers: Dict[str, List[EventSubscriber]] = {}
        self.active_pipelines: Dict[str, PipelineInfo] = {}
        self.shared_spaces: Dict[str, SharedMemorySpace] = {}
        
    async def register_pipeline(self, pipeline_id: str, info: PipelineInfo):
        """Register a pipeline for communication."""
        self.active_pipelines[pipeline_id] = info
        await self._notify_pipeline_registration(pipeline_id)
    
    async def send_message(
        self, 
        from_pipeline: str, 
        to_pipeline: str, 
        message: Any,
        message_type: str = "data"
    ):
        """Send direct message between pipelines."""
        message_envelope = PipelineMessage(
            from_pipeline=from_pipeline,
            to_pipeline=to_pipeline,
            message=message,
            message_type=message_type,
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4())
        )
        
        await self.message_queue.put(message_envelope)
        await self._log_message(message_envelope)
    
    async def broadcast_event(self, event_type: str, data: Any, source_pipeline: str):
        """Broadcast event to all subscribers."""
        if event_type in self.event_subscribers:
            event = PipelineEvent(
                event_type=event_type,
                data=data,
                source_pipeline=source_pipeline,
                timestamp=datetime.now(),
                event_id=str(uuid.uuid4())
            )
            
            # Send to all subscribers
            for subscriber in self.event_subscribers[event_type]:
                await subscriber.handle_event(event)
    
    def subscribe_to_events(
        self, 
        pipeline_id: str, 
        event_types: List[str], 
        callback: Callable
    ):
        """Subscribe to events from other pipelines."""
        subscriber = EventSubscriber(pipeline_id, callback)
        
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = []
            self.event_subscribers[event_type].append(subscriber)
```

#### Shared Memory Spaces
```python
class SharedMemorySpace:
    """Shared memory space for related pipelines."""
    
    def __init__(self, space_id: str):
        self.space_id = space_id
        self.data: Dict[str, Any] = {}
        self.access_log: List[Dict[str, Any]] = []
        self.locks: Dict[str, asyncio.Lock] = {}
        
    async def write(self, key: str, value: Any, pipeline_id: str):
        """Write data to shared space."""
        async with self._get_lock(key):
            self.data[key] = value
            self._log_access("write", key, pipeline_id)
    
    async def read(self, key: str, pipeline_id: str) -> Any:
        """Read data from shared space."""
        async with self._get_lock(key):
            value = self.data.get(key)
            self._log_access("read", key, pipeline_id)
            return value
    
    async def delete(self, key: str, pipeline_id: str):
        """Delete data from shared space."""
        async with self._get_lock(key):
            if key in self.data:
                del self.data[key]
                self._log_access("delete", key, pipeline_id)
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for key."""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]
```

### Acceptance Criteria
- [ ] Message delivery is reliable and ordered
- [ ] Event broadcasting works correctly
- [ ] Shared memory spaces are thread-safe
- [ ] Communication is properly logged
- [ ] Performance is acceptable under load

---

## Integration Testing Requirements

### Cross-Task Integration
- Pipeline executor uses step executor correctly
- Agent interface properly routes pipeline vs single-goal execution
- Communication system works with executor
- Memory persistence is maintained across all components

### Performance Testing
- Concurrent step execution scales appropriately
- Memory usage stays within acceptable limits
- Communication latency is minimal
- Error recovery doesn't impact performance

### Error Handling Testing
- Step failures are handled gracefully
- Pipeline communication failures are recoverable
- Timeout handling works correctly
- Retry logic prevents infinite loops

## Success Metrics
- All phase 2 tasks completed with high test coverage
- Integration with phase 1 components works seamlessly
- Performance benchmarks are met
- Error handling is comprehensive
- Documentation is complete and accurate
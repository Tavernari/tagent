# Phase 3: Enhanced Features Tasks

## Task 3.1: Conditional Execution (`src/tagent/pipeline/conditions.py`)
**Priority: MEDIUM** | **Estimated Time: 4-5 days**

### Objective
Implement conditional execution logic that allows steps to execute based on results from previous steps.

### Requirements
- [x] Create condition evaluation engine
- [x] Add support for result-based conditions
- [x] Implement dynamic step enabling/disabling
- [x] Add condition DSL (Domain Specific Language)
- [x] Create condition validation logic

### Implementation Details

#### Condition Evaluation Engine
```python
class ConditionEvaluator:
    """Evaluates conditions for conditional step execution."""
    
    def __init__(self):
        self.operators = {
            'equals': self._equals,
            'not_equals': self._not_equals,
            'contains': self._contains,
            'greater_than': self._greater_than,
            'less_than': self._less_than,
            'exists': self._exists,
            'not_exists': self._not_exists,
            'and': self._and,
            'or': self._or
        }
    
    def evaluate(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a condition against the current context."""
        if not isinstance(condition, dict):
            return bool(condition)
        
        operator = condition.get('operator')
        if operator not in self.operators:
            raise ValueError(f"Unknown operator: {operator}")
        
        return self.operators[operator](condition, context)
    
    def _equals(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if two values are equal."""
        left = self._resolve_value(condition.get('left'), context)
        right = self._resolve_value(condition.get('right'), context)
        return left == right
    
    def _contains(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if value contains substring or item."""
        container = self._resolve_value(condition.get('container'), context)
        item = self._resolve_value(condition.get('item'), context)
        
        if isinstance(container, str):
            return str(item) in container
        elif isinstance(container, (list, tuple)):
            return item in container
        elif isinstance(container, dict):
            return item in container
        
        return False
    
    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        """Resolve a value that might be a reference to context data."""
        if isinstance(value, str) and value.startswith('$'):
            # Reference to context data
            key = value[1:]  # Remove $ prefix
            return self._get_nested_value(context, key)
        return value
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
```

#### Conditional Step Enhancement
```python
@dataclass
class ConditionalStep(PipelineStep):
    """Enhanced pipeline step with conditional execution."""
    condition: Optional[Dict[str, Any]] = None
    condition_mode: str = "pre"  # "pre" or "post"
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        """Check if step should execute based on condition."""
        if not self.condition:
            return True
        
        if self.condition_mode == "pre":
            evaluator = ConditionEvaluator()
            return evaluator.evaluate(self.condition, context)
        
        return True  # Post-conditions evaluated after execution
```

#### Condition DSL
```python
class ConditionDSL:
    """Domain Specific Language for pipeline conditions."""
    
    @staticmethod
    def step_result_contains(step_name: str, value: Any) -> Dict[str, Any]:
        """Create condition: step result contains value."""
        return {
            'operator': 'contains',
            'container': f'${step_name}.result',
            'item': value
        }
    
    @staticmethod
    def step_succeeded(step_name: str) -> Dict[str, Any]:
        """Create condition: step succeeded."""
        return {
            'operator': 'equals',
            'left': f'${step_name}.status',
            'right': 'success'
        }
    
    @staticmethod
    def data_exists(key: str) -> Dict[str, Any]:
        """Create condition: data exists in context."""
        return {
            'operator': 'exists',
            'key': f'${key}'
        }
    
    @staticmethod
    def combine_and(*conditions) -> Dict[str, Any]:
        """Combine conditions with AND logic."""
        return {
            'operator': 'and',
            'conditions': list(conditions)
        }
    
    @staticmethod
    def combine_or(*conditions) -> Dict[str, Any]:
        """Combine conditions with OR logic."""
        return {
            'operator': 'or',
            'conditions': list(conditions)
        }

# Usage example
condition = ConditionDSL.combine_and(
    ConditionDSL.step_succeeded("competitor_analysis"),
    ConditionDSL.step_result_contains("competitor_analysis", "market_gap")
)
```

### Example Usage
```python
# Pipeline with conditional execution
pipeline = Pipeline("conditional_analysis")

pipeline.step(
    name="data_collection",
    goal="Collect market data"
).step(
    name="analysis",
    goal="Analyze collected data",
    depends_on=["data_collection"]
).step(
    name="deep_analysis",
    goal="Perform deep analysis if gaps found",
    depends_on=["analysis"],
    condition=ConditionDSL.step_result_contains("analysis", "requires_deep_analysis"),
    execution_mode=ExecutionMode.CONDITIONAL
).step(
    name="report",
    goal="Generate final report",
    depends_on=["analysis", "deep_analysis"],
    condition=ConditionDSL.combine_or(
        ConditionDSL.step_succeeded("analysis"),
        ConditionDSL.step_succeeded("deep_analysis")
    )
)
```

### Acceptance Criteria
- [x] Conditions evaluate correctly with various operators
- [x] DSL provides intuitive condition creation
- [x] Conditional steps skip execution when conditions fail
- [x] Complex conditions with AND/OR work properly
- [x] Condition validation prevents invalid configurations

---

## Task 3.2: Pipeline Persistence (`src/tagent/pipeline/persistence.py`)
**Priority: MEDIUM** | **Estimated Time: 5-6 days**

### Objective
Implement comprehensive pipeline persistence for state recovery, history tracking, and audit trails.

### Requirements
- [x] Implement pipeline state save/load
- [x] Add checkpoint creation at key points
- [x] Create execution history tracking
- [x] Add audit trail functionality
- [x] Implement state cleanup mechanisms

### Implementation Details

#### Pipeline State Persistence
```python
class PipelinePersistenceManager:
    """Manages pipeline state persistence and recovery."""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
        self.checkpoint_manager = CheckpointManager(storage_backend)
        self.history_manager = HistoryManager(storage_backend)
        self.audit_manager = AuditManager(storage_backend)
    
    async def save_pipeline_state(self, pipeline_id: str, state: PipelineState):
        """Save complete pipeline state."""
        state_data = {
            'pipeline_id': pipeline_id,
            'state': state.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        await self.storage.save(f"pipeline_state_{pipeline_id}", state_data)
        await self.audit_manager.log_event(
            'state_saved',
            pipeline_id,
            {'state_size': len(json.dumps(state_data))}
        )
    
    async def load_pipeline_state(self, pipeline_id: str) -> Optional[PipelineState]:
        """Load pipeline state from storage."""
        try:
            state_data = await self.storage.load(f"pipeline_state_{pipeline_id}")
            if state_data:
                state = PipelineState.from_dict(state_data['state'])
                await self.audit_manager.log_event('state_loaded', pipeline_id)
                return state
        except Exception as e:
            await self.audit_manager.log_event(
                'state_load_failed',
                pipeline_id,
                {'error': str(e)}
            )
        return None
    
    async def create_checkpoint(self, pipeline_id: str, checkpoint_type: str):
        """Create a checkpoint of current pipeline state."""
        await self.checkpoint_manager.create_checkpoint(
            pipeline_id,
            checkpoint_type
        )
    
    async def restore_from_checkpoint(
        self, 
        pipeline_id: str, 
        checkpoint_id: str
    ) -> PipelineState:
        """Restore pipeline state from checkpoint."""
        return await self.checkpoint_manager.restore_checkpoint(
            pipeline_id,
            checkpoint_id
        )
```

#### Checkpoint Management
```python
class CheckpointManager:
    """Manages pipeline checkpoints for recovery."""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
    
    async def create_checkpoint(
        self, 
        pipeline_id: str, 
        checkpoint_type: str,
        metadata: Dict[str, Any] = None
    ):
        """Create a checkpoint with metadata."""
        checkpoint_id = f"{pipeline_id}_{checkpoint_type}_{int(time.time())}"
        
        # Load current state
        current_state = await self.storage.load(f"pipeline_state_{pipeline_id}")
        
        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'pipeline_id': pipeline_id,
            'checkpoint_type': checkpoint_type,
            'state': current_state,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        await self.storage.save(f"checkpoint_{checkpoint_id}", checkpoint_data)
        
        # Update checkpoint index
        await self._update_checkpoint_index(pipeline_id, checkpoint_id)
    
    async def restore_checkpoint(
        self, 
        pipeline_id: str, 
        checkpoint_id: str
    ) -> PipelineState:
        """Restore state from specific checkpoint."""
        checkpoint_data = await self.storage.load(f"checkpoint_{checkpoint_id}")
        
        if not checkpoint_data:
            raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")
        
        # Restore state
        state = PipelineState.from_dict(checkpoint_data['state'])
        
        # Save restored state as current
        await self.storage.save(f"pipeline_state_{pipeline_id}", checkpoint_data['state'])
        
        return state
    
    async def list_checkpoints(self, pipeline_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a pipeline."""
        index_data = await self.storage.load(f"checkpoint_index_{pipeline_id}")
        return index_data.get('checkpoints', []) if index_data else []
```

#### Execution History Tracking
```python
class HistoryManager:
    """Tracks and manages pipeline execution history."""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
    
    async def record_execution_start(self, pipeline_id: str, config: Dict[str, Any]):
        """Record the start of pipeline execution."""
        execution_id = str(uuid.uuid4())
        
        history_entry = {
            'execution_id': execution_id,
            'pipeline_id': pipeline_id,
            'event_type': 'execution_start',
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'status': 'running'
        }
        
        await self._add_history_entry(pipeline_id, history_entry)
        return execution_id
    
    async def record_step_completion(
        self, 
        pipeline_id: str, 
        execution_id: str,
        step_name: str,
        result: Any,
        duration: float
    ):
        """Record completion of a pipeline step."""
        history_entry = {
            'execution_id': execution_id,
            'pipeline_id': pipeline_id,
            'event_type': 'step_completed',
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'result_size': len(json.dumps(result)) if result else 0,
            'status': 'success'
        }
        
        await self._add_history_entry(pipeline_id, history_entry)
    
    async def record_execution_end(
        self, 
        pipeline_id: str, 
        execution_id: str,
        status: str,
        final_result: Any = None
    ):
        """Record the end of pipeline execution."""
        history_entry = {
            'execution_id': execution_id,
            'pipeline_id': pipeline_id,
            'event_type': 'execution_end',
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'final_result': final_result
        }
        
        await self._add_history_entry(pipeline_id, history_entry)
    
    async def get_execution_history(
        self, 
        pipeline_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history for a pipeline."""
        history_data = await self.storage.load(f"history_{pipeline_id}")
        
        if not history_data:
            return []
        
        entries = history_data.get('entries', [])
        return sorted(entries, key=lambda x: x['timestamp'], reverse=True)[:limit]
```

### Acceptance Criteria
- [x] State persistence survives system restarts
- [x] Checkpoints enable recovery from failures
- [x] History tracking provides complete audit trail
- [x] Cleanup mechanisms prevent storage bloat
- [x] Performance impact is minimal

---

## Task 3.3: Pipeline Monitoring (`src/tagent/pipeline/monitoring.py`)
**Priority: MEDIUM** | **Estimated Time: 4-5 days**

### Objective
Implement comprehensive monitoring for pipeline execution with metrics, alerts, and visualization.

### Requirements
- [x] Add real-time progress tracking
- [x] Implement performance metrics collection
- [x] Create failure analysis and reporting
- [x] Add resource usage monitoring
- [x] Implement execution visualization

### Implementation Details

#### Real-time Progress Tracking
```python
class PipelineMonitor:
    """Monitors pipeline execution in real-time."""
    
    def __init__(self):
        self.active_pipelines: Dict[str, PipelineProgress] = {}
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.progress_callbacks: List[Callable] = []
    
    def start_monitoring(self, pipeline_id: str, total_steps: int):
        """Start monitoring a pipeline execution."""
        progress = PipelineProgress(
            pipeline_id=pipeline_id,
            total_steps=total_steps,
            start_time=datetime.now()
        )
        
        self.active_pipelines[pipeline_id] = progress
        self.metrics_collector.start_collection(pipeline_id)
    
    def update_step_progress(
        self, 
        pipeline_id: str, 
        step_name: str, 
        status: str,
        result: Any = None
    ):
        """Update progress for a specific step."""
        if pipeline_id not in self.active_pipelines:
            return
        
        progress = self.active_pipelines[pipeline_id]
        progress.update_step(step_name, status, result)
        
        # Collect metrics
        self.metrics_collector.record_step_event(
            pipeline_id,
            step_name,
            status,
            datetime.now()
        )
        
        # Check for alerts
        self._check_alerts(pipeline_id, progress)
        
        # Notify progress callbacks
        self._notify_progress_callbacks(pipeline_id, progress)
    
    def get_pipeline_progress(self, pipeline_id: str) -> Optional[PipelineProgress]:
        """Get current progress for a pipeline."""
        return self.active_pipelines.get(pipeline_id)
    
    def get_all_active_pipelines(self) -> Dict[str, PipelineProgress]:
        """Get progress for all active pipelines."""
        return self.active_pipelines.copy()
```

#### Performance Metrics Collection
```python
class MetricsCollector:
    """Collects and aggregates pipeline performance metrics."""
    
    def __init__(self):
        self.metrics_storage = {}
        self.real_time_metrics = {}
    
    def start_collection(self, pipeline_id: str):
        """Start collecting metrics for a pipeline."""
        self.real_time_metrics[pipeline_id] = {
            'start_time': time.time(),
            'step_durations': {},
            'resource_usage': [],
            'error_counts': 0,
            'memory_usage': []
        }
    
    def record_step_event(
        self, 
        pipeline_id: str, 
        step_name: str, 
        event_type: str,
        timestamp: datetime
    ):
        """Record a step event with metrics."""
        if pipeline_id not in self.real_time_metrics:
            return
        
        metrics = self.real_time_metrics[pipeline_id]
        
        if event_type == 'step_started':
            metrics['step_durations'][step_name] = {
                'start_time': timestamp,
                'duration': None
            }
        elif event_type == 'step_completed':
            if step_name in metrics['step_durations']:
                start_time = metrics['step_durations'][step_name]['start_time']
                duration = (timestamp - start_time).total_seconds()
                metrics['step_durations'][step_name]['duration'] = duration
        elif event_type == 'step_failed':
            metrics['error_counts'] += 1
    
    def record_resource_usage(self, pipeline_id: str):
        """Record current resource usage."""
        if pipeline_id not in self.real_time_metrics:
            return
        
        # Get system metrics
        import psutil
        
        usage = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024)
        }
        
        self.real_time_metrics[pipeline_id]['resource_usage'].append(usage)
    
    def get_pipeline_metrics(self, pipeline_id: str) -> Dict[str, Any]:
        """Get aggregated metrics for a pipeline."""
        if pipeline_id not in self.real_time_metrics:
            return {}
        
        metrics = self.real_time_metrics[pipeline_id]
        
        # Calculate aggregated metrics
        total_duration = time.time() - metrics['start_time']
        step_durations = [
            d['duration'] for d in metrics['step_durations'].values()
            if d['duration'] is not None
        ]
        
        return {
            'pipeline_id': pipeline_id,
            'total_duration': total_duration,
            'average_step_duration': sum(step_durations) / len(step_durations) if step_durations else 0,
            'max_step_duration': max(step_durations) if step_durations else 0,
            'min_step_duration': min(step_durations) if step_durations else 0,
            'error_count': metrics['error_counts'],
            'steps_completed': len(step_durations),
            'resource_usage_summary': self._summarize_resource_usage(metrics['resource_usage'])
        }
```

#### Failure Analysis and Reporting
```python
class FailureAnalyzer:
    """Analyzes pipeline failures and generates reports."""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
    
    async def analyze_failure(
        self, 
        pipeline_id: str, 
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a pipeline failure and generate report."""
        failure_report = {
            'pipeline_id': pipeline_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'analysis': self._analyze_error_pattern(error),
            'recommendations': self._generate_recommendations(error, context)
        }
        
        # Save failure report
        await self.storage.save(
            f"failure_report_{pipeline_id}_{int(time.time())}", 
            failure_report
        )
        
        return failure_report
    
    def _analyze_error_pattern(self, error: Exception) -> Dict[str, Any]:
        """Analyze error patterns for insights."""
        analysis = {
            'error_category': self._categorize_error(error),
            'is_retryable': self._is_retryable_error(error),
            'potential_causes': self._identify_potential_causes(error),
            'similar_failures': self._find_similar_failures(error)
        }
        
        return analysis
    
    def _generate_recommendations(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for preventing similar failures."""
        recommendations = []
        
        if isinstance(error, TimeoutError):
            recommendations.extend([
                "Consider increasing step timeout values",
                "Check for network connectivity issues",
                "Review LLM response times"
            ])
        elif isinstance(error, MemoryError):
            recommendations.extend([
                "Reduce batch sizes in data processing",
                "Implement data streaming for large datasets",
                "Consider distributed execution"
            ])
        elif isinstance(error, ConnectionError):
            recommendations.extend([
                "Implement retry logic with exponential backoff",
                "Check API rate limits",
                "Verify network connectivity"
            ])
        
        return recommendations
```

### Acceptance Criteria
- [x] Real-time progress tracking updates correctly
- [x] Performance metrics are accurate and useful
- [x] Failure analysis provides actionable insights
- [x] Resource monitoring detects issues early
- [x] Visualization provides clear status overview

---

## Task 3.4: Pipeline Templates (`src/tagent/pipeline/templates.py`)
**Priority: LOW** | **Estimated Time: 3-4 days**

### Objective
Create a library of reusable pipeline templates for common use cases.

### Requirements
- [x] Create template registry system
- [x] Implement template parameterization
- [x] Add template validation and testing
- [x] Create common use case templates
- [x] Add template documentation generator

### Implementation Details

#### Template Registry
```python
class PipelineTemplateRegistry:
    """Registry for reusable pipeline templates."""
    
    def __init__(self):
        self.templates: Dict[str, PipelineTemplate] = {}
        self._load_builtin_templates()
    
    def register_template(self, template: PipelineTemplate):
        """Register a new template."""
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PipelineTemplate]:
        """Get template by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self.templates.keys())
    
    def create_pipeline_from_template(
        self, 
        template_name: str, 
        parameters: Dict[str, Any]
    ) -> Pipeline:
        """Create pipeline instance from template."""
        template = self.get_template(template_name)
        if not template:
            raise TemplateNotFoundError(f"Template '{template_name}' not found")
        
        return template.create_pipeline(parameters)
```

#### Common Templates
```python
# Company Research Template
COMPANY_RESEARCH_TEMPLATE = PipelineTemplate(
    name="company_research",
    description="Comprehensive company analysis pipeline",
    parameters={
        "company_name": {"type": "string", "required": True},
        "include_social_media": {"type": "boolean", "default": True},
        "include_reviews": {"type": "boolean", "default": True},
        "report_format": {"type": "string", "default": "markdown"}
    },
    pipeline_factory=lambda params: Pipeline("company_research")
        .step(
            name="web_search",
            goal=f"Search for information about {params['company_name']}",
            constraints=["Use reliable sources", "Focus on recent information"]
        )
        .step(
            name="review_analysis",
            goal=f"Analyze customer reviews for {params['company_name']}",
            depends_on=["web_search"],
            condition=ConditionDSL.equals("$include_reviews", True)
        )
        .step(
            name="social_media_analysis",
            goal=f"Analyze social media presence of {params['company_name']}",
            depends_on=["web_search"],
            condition=ConditionDSL.equals("$include_social_media", True),
            execution_mode=ExecutionMode.CONCURRENT
        )
        .step(
            name="final_report",
            goal=f"Generate comprehensive report in {params['report_format']} format",
            depends_on=["web_search", "review_analysis", "social_media_analysis"]
        )
)

# E-commerce Analysis Template
ECOMMERCE_ANALYSIS_TEMPLATE = PipelineTemplate(
    name="ecommerce_analysis",
    description="E-commerce product and market analysis",
    parameters={
        "product_category": {"type": "string", "required": True},
        "target_market": {"type": "string", "required": True},
        "budget_range": {"type": "string", "default": "all"}
    },
    pipeline_factory=lambda params: Pipeline("ecommerce_analysis")
        .step(
            name="product_research",
            goal=f"Research {params['product_category']} products in {params['target_market']}",
            tools_filter=["web_search", "product_search"]
        )
        .step(
            name="competitor_analysis",
            goal=f"Analyze competitors in {params['product_category']} market",
            depends_on=["product_research"],
            execution_mode=ExecutionMode.CONCURRENT
        )
        .step(
            name="market_analysis",
            goal=f"Analyze market trends for {params['product_category']}",
            depends_on=["product_research"],
            execution_mode=ExecutionMode.CONCURRENT
        )
        .step(
            name="opportunity_analysis",
            goal="Identify market opportunities and gaps",
            depends_on=["competitor_analysis", "market_analysis"],
            condition=ConditionDSL.combine_and(
                ConditionDSL.step_succeeded("competitor_analysis"),
                ConditionDSL.step_succeeded("market_analysis")
            )
        )
)
```

### Acceptance Criteria
- [x] Template registry manages templates correctly
- [x] Templates can be parameterized effectively
- [x] Common use case templates are available
- [x] Template validation ensures correctness
- [x] Documentation is generated automatically

---

## Integration Requirements

### Cross-Phase Dependencies
- Enhanced features build on Phase 1 and Phase 2 infrastructure
- Conditional execution integrates with step execution
- Monitoring uses persistence for historical data
- Templates use all previous components

### Testing Requirements
- Unit tests for each enhanced feature
- Integration tests with core system
- Performance impact assessment
- User acceptance testing

### Documentation Requirements
- Feature-specific documentation
- Integration guides
- Performance considerations
- Best practices

## Success Metrics
- All enhanced features work correctly
- Performance impact is minimal
- User experience is improved
- System reliability is maintained
- Documentation is comprehensive
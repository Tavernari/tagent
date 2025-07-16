# Phase 4: Integration & API Tasks

## Task 4.1: Pipeline Builder API (`src/tagent/pipeline/api.py`)
**Priority: MEDIUM** | **Estimated Time: 4-5 days**

### Objective
Create a comprehensive API for pipeline construction, validation, and management.

### Requirements
- [ ] Create fluent interface for pipeline construction
- [ ] Add pipeline validation and optimization
- [ ] Implement template pipeline library
- [ ] Add pipeline import/export functionality
- [ ] Create pipeline composition utilities

### Implementation Details

#### Fluent Pipeline Builder
```python
class PipelineBuilder:
    """Fluent interface for building pipelines."""
    
    def __init__(self, name: str, description: str = ""):
        self.pipeline = Pipeline(name, description)
        self.validation_errors: List[str] = []
        
    def step(self, name: str, goal: str, **kwargs) -> 'PipelineBuilder':
        """Add a step to the pipeline."""
        step = PipelineStep(name=name, goal=goal, **kwargs)
        self.pipeline.steps.append(step)
        return self
    
    def parallel_steps(self, *step_configs) -> 'PipelineBuilder':
        """Add multiple steps that can run in parallel."""
        for config in step_configs:
            config.setdefault('execution_mode', ExecutionMode.CONCURRENT)
            self.step(**config)
        return self
    
    def conditional_step(self, name: str, goal: str, condition: Dict[str, Any], **kwargs) -> 'PipelineBuilder':
        """Add a conditional step."""
        kwargs['condition'] = condition
        kwargs['execution_mode'] = ExecutionMode.CONDITIONAL
        return self.step(name, goal, **kwargs)
    
    def add_global_constraint(self, constraint: str) -> 'PipelineBuilder':
        """Add a global constraint to the pipeline."""
        self.pipeline.global_constraints.append(constraint)
        return self
    
    def set_shared_data(self, key: str, value: Any) -> 'PipelineBuilder':
        """Set shared data for the pipeline."""
        self.pipeline.shared_context[key] = value
        return self
    
    def validate(self) -> 'PipelineBuilder':
        """Validate the pipeline configuration."""
        self.validation_errors = self.pipeline.validate()
        
        # Additional validation
        self._validate_step_dependencies()
        self._validate_execution_modes()
        self._validate_conditions()
        
        return self
    
    def build(self) -> Pipeline:
        """Build and return the final pipeline."""
        # Validate before building
        self.validate()
        
        if self.validation_errors:
            raise PipelineValidationError(
                f"Pipeline validation failed: {', '.join(self.validation_errors)}"
            )
        
        # Optimize pipeline
        self._optimize_pipeline()
        
        return self.pipeline
    
    def _validate_step_dependencies(self):
        """Validate step dependencies."""
        step_names = {step.name for step in self.pipeline.steps}
        
        for step in self.pipeline.steps:
            for dep in step.depends_on:
                if dep not in step_names:
                    self.validation_errors.append(
                        f"Step '{step.name}' depends on unknown step '{dep}'"
                    )
    
    def _optimize_pipeline(self):
        """Optimize pipeline for better performance."""
        # Analyze dependency graph for optimization opportunities
        optimizer = PipelineOptimizer(self.pipeline)
        optimized_steps = optimizer.optimize_execution_order()
        
        # Apply optimizations
        self.pipeline.steps = optimized_steps
```

#### Pipeline Optimization
```python
class PipelineOptimizer:
    """Optimizes pipeline execution for better performance."""
    
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.dependency_graph = self._build_dependency_graph()
    
    def optimize_execution_order(self) -> List[PipelineStep]:
        """Optimize step execution order for maximum parallelism."""
        # Group steps by dependency levels
        levels = self._calculate_dependency_levels()
        
        # Optimize within each level
        optimized_steps = []
        for level in levels:
            # Sort by estimated execution time (if available)
            level_steps = sorted(level, key=self._estimate_execution_time)
            optimized_steps.extend(level_steps)
        
        return optimized_steps
    
    def identify_parallel_opportunities(self) -> List[List[PipelineStep]]:
        """Identify steps that can be executed in parallel."""
        parallel_groups = []
        
        # Find steps with no interdependencies
        for step in self.pipeline.steps:
            candidates = self._find_parallel_candidates(step)
            if candidates:
                parallel_groups.append(candidates)
        
        return parallel_groups
    
    def _calculate_dependency_levels(self) -> List[List[PipelineStep]]:
        """Calculate dependency levels for steps."""
        levels = []
        processed = set()
        
        while len(processed) < len(self.pipeline.steps):
            current_level = []
            
            for step in self.pipeline.steps:
                if step.name in processed:
                    continue
                
                # Check if all dependencies are processed
                if all(dep in processed for dep in step.depends_on):
                    current_level.append(step)
            
            if not current_level:
                break  # Circular dependency or other issue
            
            levels.append(current_level)
            processed.update(step.name for step in current_level)
        
        return levels
```

#### Pipeline Import/Export
```python
class PipelineSerializer:
    """Handles pipeline serialization and deserialization."""
    
    def __init__(self):
        self.supported_formats = ['json', 'yaml', 'pickle']
    
    def export_pipeline(self, pipeline: Pipeline, format: str = 'json') -> str:
        """Export pipeline to specified format."""
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        pipeline_data = self._pipeline_to_dict(pipeline)
        
        if format == 'json':
            return json.dumps(pipeline_data, indent=2, default=self._json_serializer)
        elif format == 'yaml':
            import yaml
            return yaml.dump(pipeline_data, default_flow_style=False)
        elif format == 'pickle':
            import pickle
            return pickle.dumps(pipeline_data)
    
    def import_pipeline(self, data: str, format: str = 'json') -> Pipeline:
        """Import pipeline from specified format."""
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        if format == 'json':
            pipeline_data = json.loads(data)
        elif format == 'yaml':
            import yaml
            pipeline_data = yaml.safe_load(data)
        elif format == 'pickle':
            import pickle
            pipeline_data = pickle.loads(data)
        
        return self._dict_to_pipeline(pipeline_data)
    
    def _pipeline_to_dict(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            'name': pipeline.name,
            'description': pipeline.description,
            'steps': [self._step_to_dict(step) for step in pipeline.steps],
            'global_constraints': pipeline.global_constraints,
            'shared_context': pipeline.shared_context,
            'version': '1.0'
        }
    
    def _step_to_dict(self, step: PipelineStep) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        return {
            'name': step.name,
            'goal': step.goal,
            'constraints': step.constraints,
            'depends_on': step.depends_on,
            'execution_mode': step.execution_mode.value,
            'timeout': step.timeout,
            'max_retries': step.max_retries,
            'tools_filter': step.tools_filter,
            'condition': getattr(step, 'condition', None)
        }
```

### Example Usage
```python
# Using the fluent builder API
pipeline = (PipelineBuilder("advanced_research", "Advanced research pipeline")
    .step(
        name="initial_search",
        goal="Perform initial research",
        constraints=["Use reliable sources"]
    )
    .parallel_steps(
        {
            "name": "deep_analysis",
            "goal": "Perform deep analysis",
            "depends_on": ["initial_search"],
            "timeout": 300
        },
        {
            "name": "competitor_check",
            "goal": "Check competitors",
            "depends_on": ["initial_search"],
            "tools_filter": ["web_search"]
        }
    )
    .conditional_step(
        name="additional_research",
        goal="Perform additional research if needed",
        condition=ConditionDSL.step_result_contains("deep_analysis", "needs_more_data"),
        depends_on=["deep_analysis"]
    )
    .step(
        name="final_report",
        goal="Generate comprehensive report",
        depends_on=["deep_analysis", "competitor_check", "additional_research"]
    )
    .add_global_constraint("Complete within 30 minutes")
    .set_shared_data("research_depth", "comprehensive")
    .validate()
    .build()
)

# Export pipeline
serializer = PipelineSerializer()
pipeline_json = serializer.export_pipeline(pipeline, 'json')

# Import pipeline
restored_pipeline = serializer.import_pipeline(pipeline_json, 'json')
```

### Acceptance Criteria
- [ ] Fluent interface is intuitive and easy to use
- [ ] Pipeline validation catches all configuration errors
- [ ] Optimization improves execution performance
- [ ] Import/export maintains pipeline integrity
- [ ] API documentation is comprehensive

---

## Task 4.2: Package Configuration (`pyproject.toml` and imports)
**Priority: HIGH** | **Estimated Time: 2-3 days**

### Objective
Configure package structure with optional dependencies and graceful fallbacks.

### Requirements
- [ ] Update `pyproject.toml` with optional dependencies
- [ ] Add pipeline extras configuration
- [ ] Update package imports with graceful fallbacks
- [ ] Create installation documentation
- [ ] Add version compatibility checks

### Implementation Details

#### Package Configuration
```toml
[project.optional-dependencies]
# Core pipeline functionality
pipeline = [
    "asyncio-throttle>=1.0.0",  # Concurrent execution throttling
    "networkx>=2.5.0",          # Dependency graph management
]

# Enhanced monitoring and metrics
monitoring = [
    "psutil>=5.8.0",            # System resource monitoring
    "prometheus-client>=0.15.0", # Metrics collection
]

# Persistence and storage
persistence = [
    "sqlalchemy>=1.4.0",        # Database persistence
    "alembic>=1.7.0",           # Database migrations
    "redis>=4.0.0",             # Redis storage backend
]

# Web UI and visualization
ui = [
    "streamlit>=1.28.0",        # Web-based pipeline builder
    "plotly>=5.0.0",            # Execution visualization
    "graphviz>=0.20.0",         # Pipeline graph visualization
]

# All pipeline features
all = [
    "tagent[pipeline]",
    "tagent[monitoring]", 
    "tagent[persistence]",
    "tagent[ui]",
]
```

#### Graceful Import Handling
```python
# src/tagent/pipeline/__init__.py
"""
TAgent Pipeline System - Optional pipeline functionality.

This module provides advanced pipeline orchestration capabilities.
Requires: pip install tagent[pipeline]
"""

# Core pipeline imports
try:
    from .models import Pipeline, PipelineStep, PipelineResult, ExecutionMode
    from .executor import PipelineExecutor
    from .scheduler import PipelineScheduler
    from .api import PipelineBuilder
    PIPELINE_CORE_AVAILABLE = True
except ImportError as e:
    PIPELINE_CORE_AVAILABLE = False
    _CORE_IMPORT_ERROR = str(e)

# Monitoring imports
try:
    from .monitoring import PipelineMonitor, MetricsCollector
    PIPELINE_MONITORING_AVAILABLE = True
except ImportError:
    PIPELINE_MONITORING_AVAILABLE = False

# Persistence imports
try:
    from .persistence import PipelinePersistenceManager
    PIPELINE_PERSISTENCE_AVAILABLE = True
except ImportError:
    PIPELINE_PERSISTENCE_AVAILABLE = False

# UI imports
try:
    from .ui import PipelineVisualizer
    PIPELINE_UI_AVAILABLE = True
except ImportError:
    PIPELINE_UI_AVAILABLE = False

# Feature availability check
def check_feature_availability():
    """Check which pipeline features are available."""
    return {
        'core': PIPELINE_CORE_AVAILABLE,
        'monitoring': PIPELINE_MONITORING_AVAILABLE,
        'persistence': PIPELINE_PERSISTENCE_AVAILABLE,
        'ui': PIPELINE_UI_AVAILABLE
    }

def require_feature(feature_name: str):
    """Require a specific feature to be available."""
    availability = check_feature_availability()
    
    if not availability.get(feature_name, False):
        feature_extras = {
            'core': 'pipeline',
            'monitoring': 'monitoring',
            'persistence': 'persistence',
            'ui': 'ui'
        }
        
        extra = feature_extras.get(feature_name, feature_name)
        raise ImportError(
            f"Pipeline {feature_name} features are not available. "
            f"Install with: pip install tagent[{extra}]"
        )

# Dummy classes for type hints when not available
if not PIPELINE_CORE_AVAILABLE:
    class Pipeline:
        def __init__(self, *args, **kwargs):
            require_feature('core')
    
    class PipelineStep:
        def __init__(self, *args, **kwargs):
            require_feature('core')
    
    class PipelineResult:
        def __init__(self, *args, **kwargs):
            require_feature('core')
    
    class ExecutionMode:
        SERIAL = "serial"
        CONCURRENT = "concurrent"
        CONDITIONAL = "conditional"
```

#### Enhanced Agent Integration
```python
# src/tagent/agent.py - Enhanced version
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline, PipelineResult

def run_agent(
    goal_or_pipeline: Union[str, 'Pipeline'],
    config: Optional['TAgentConfig'] = None,
    **kwargs
) -> Union[TaskBasedAgentResult, 'PipelineResult']:
    """
    Enhanced run_agent supporting both single goals and pipelines.
    
    Args:
        goal_or_pipeline: Either a goal string or Pipeline object
        config: Optional configuration object
        **kwargs: Additional parameters for backward compatibility
        
    Returns:
        TaskBasedAgentResult for single goals or PipelineResult for pipelines
    """
    # Check if input is a Pipeline object
    if hasattr(goal_or_pipeline, 'steps') and hasattr(goal_or_pipeline, 'name'):
        # Import pipeline functionality
        try:
            from .pipeline import run_pipeline
            return asyncio.run(run_pipeline(goal_or_pipeline, config, **kwargs))
        except ImportError:
            raise ImportError(
                "Pipeline support requires additional dependencies. "
                "Install with: pip install tagent[pipeline]"
            )
    else:
        # Existing single-goal execution
        return run_task_based_agent(goal_or_pipeline, config, **kwargs)
```

### Acceptance Criteria
- [ ] Package installs correctly with different extra combinations
- [ ] Graceful fallbacks work when features are not available
- [ ] Error messages are clear and actionable
- [ ] Type hints work correctly in all scenarios
- [ ] Backward compatibility is maintained

---

## Task 4.3: Testing & Documentation (`tests/` and `docs/`)
**Priority: HIGH** | **Estimated Time: 5-6 days**

### Objective
Create comprehensive testing suite and documentation for the pipeline system.

### Requirements
- [ ] Create unit tests for each pipeline component
- [ ] Add integration tests with existing agent system
- [ ] Create performance benchmarks
- [ ] Write comprehensive documentation
- [ ] Add example pipeline templates

### Implementation Details

#### Unit Testing Structure
```python
# tests/test_pipeline_models.py
import pytest
from src.tagent.pipeline.models import Pipeline, PipelineStep, ExecutionMode

class TestPipelineStep:
    def test_step_creation(self):
        step = PipelineStep(
            name="test_step",
            goal="Test step goal",
            depends_on=["previous_step"]
        )
        assert step.name == "test_step"
        assert step.goal == "Test step goal"
        assert step.depends_on == ["previous_step"]
    
    def test_step_validation(self):
        with pytest.raises(ValueError):
            PipelineStep(name="", goal="Test goal")
        
        with pytest.raises(ValueError):
            PipelineStep(name="test", goal="")

class TestPipeline:
    def test_pipeline_creation(self):
        pipeline = Pipeline("test_pipeline", "Test description")
        assert pipeline.name == "test_pipeline"
        assert pipeline.description == "Test description"
        assert len(pipeline.steps) == 0
    
    def test_fluent_interface(self):
        pipeline = (Pipeline("test", "Test pipeline")
            .step("step1", "First step")
            .step("step2", "Second step", depends_on=["step1"])
        )
        
        assert len(pipeline.steps) == 2
        assert pipeline.steps[1].depends_on == ["step1"]
    
    def test_pipeline_validation(self):
        pipeline = Pipeline("test", "Test pipeline")
        pipeline.step("step1", "First step")
        pipeline.step("step2", "Second step", depends_on=["nonexistent"])
        
        errors = pipeline.validate()
        assert len(errors) > 0
        assert "unknown step" in errors[0].lower()
```

#### Integration Testing
```python
# tests/test_pipeline_integration.py
import pytest
import asyncio
from src.tagent.pipeline import Pipeline, PipelineExecutor
from src.tagent.pipeline.models import ExecutionMode

class TestPipelineIntegration:
    @pytest.fixture
    def simple_pipeline(self):
        return (Pipeline("integration_test", "Test pipeline")
            .step("step1", "First step")
            .step("step2", "Second step", depends_on=["step1"])
            .step("step3", "Third step", depends_on=["step2"])
        )
    
    @pytest.fixture
    def concurrent_pipeline(self):
        return (Pipeline("concurrent_test", "Concurrent test pipeline")
            .step("initial", "Initial step")
            .step("parallel1", "Parallel step 1", 
                  depends_on=["initial"], 
                  execution_mode=ExecutionMode.CONCURRENT)
            .step("parallel2", "Parallel step 2", 
                  depends_on=["initial"], 
                  execution_mode=ExecutionMode.CONCURRENT)
            .step("final", "Final step", depends_on=["parallel1", "parallel2"])
        )
    
    @pytest.mark.asyncio
    async def test_simple_pipeline_execution(self, simple_pipeline):
        """Test basic pipeline execution."""
        executor = PipelineExecutor(simple_pipeline, {})
        result = await executor.execute()
        
        assert result.success
        assert len(result.step_results) == 3
        assert "step1" in result.step_results
        assert "step2" in result.step_results
        assert "step3" in result.step_results
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_execution(self, concurrent_pipeline):
        """Test concurrent step execution."""
        executor = PipelineExecutor(concurrent_pipeline, {})
        result = await executor.execute()
        
        assert result.success
        assert len(result.step_results) == 4
        
        # Check that parallel steps were executed concurrently
        parallel1_time = result.step_results["parallel1"]["execution_time"]
        parallel2_time = result.step_results["parallel2"]["execution_time"]
        
        # They should have started around the same time
        assert abs(parallel1_time - parallel2_time) < 1.0  # Within 1 second
```

#### Performance Benchmarks
```python
# tests/test_pipeline_performance.py
import pytest
import time
import asyncio
from src.tagent.pipeline import Pipeline, PipelineExecutor

class TestPipelinePerformance:
    @pytest.mark.benchmark
    def test_pipeline_creation_performance(self, benchmark):
        """Benchmark pipeline creation time."""
        def create_large_pipeline():
            pipeline = Pipeline("large_pipeline", "Large test pipeline")
            for i in range(100):
                pipeline.step(f"step_{i}", f"Step {i} goal")
            return pipeline
        
        result = benchmark(create_large_pipeline)
        assert len(result.steps) == 100
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_pipeline_execution_performance(self, benchmark):
        """Benchmark pipeline execution time."""
        pipeline = Pipeline("perf_test", "Performance test pipeline")
        
        # Create pipeline with multiple concurrent steps
        pipeline.step("initial", "Initial step")
        
        for i in range(10):
            pipeline.step(
                f"concurrent_{i}", 
                f"Concurrent step {i}",
                depends_on=["initial"],
                execution_mode=ExecutionMode.CONCURRENT
            )
        
        pipeline.step("final", "Final step", depends_on=[f"concurrent_{i}" for i in range(10)])
        
        async def execute_pipeline():
            executor = PipelineExecutor(pipeline, {"max_concurrent_steps": 5})
            return await executor.execute()
        
        result = await benchmark(execute_pipeline)
        assert result.success
        assert len(result.step_results) == 12  # 1 initial + 10 concurrent + 1 final
```

#### Documentation Structure
```markdown
# Pipeline System Documentation

## Overview
The TAgent Pipeline System provides advanced workflow orchestration capabilities for complex multi-step AI agent execution.

## Installation

### Basic Installation
```bash
pip install tagent[pipeline]
```

### Full Installation
```bash
pip install tagent[all]
```

## Quick Start

### Creating a Simple Pipeline
```python
from tagent.pipeline import Pipeline

pipeline = (Pipeline("example", "Example pipeline")
    .step("search", "Search for information")
    .step("analyze", "Analyze the results", depends_on=["search"])
    .step("report", "Generate report", depends_on=["analyze"])
)

result = run_agent(pipeline)
```

### Concurrent Execution
```python
from tagent.pipeline import Pipeline, ExecutionMode

pipeline = (Pipeline("concurrent_example", "Concurrent execution example")
    .step("initial", "Initial data collection")
    .step("analysis_1", "First analysis", 
          depends_on=["initial"], 
          execution_mode=ExecutionMode.CONCURRENT)
    .step("analysis_2", "Second analysis", 
          depends_on=["initial"], 
          execution_mode=ExecutionMode.CONCURRENT)
    .step("summary", "Combine results", depends_on=["analysis_1", "analysis_2"])
)
```

## API Reference

### Pipeline Class
The main class for defining pipeline workflows.

#### Methods
- `step(name, goal, **kwargs)`: Add a step to the pipeline
- `validate()`: Validate pipeline configuration
- `to_dict()`: Convert pipeline to dictionary representation

### PipelineStep Class
Represents an individual step in a pipeline.

#### Parameters
- `name`: Unique step identifier
- `goal`: Step objective description
- `depends_on`: List of step names this step depends on
- `execution_mode`: How the step should be executed
- `timeout`: Maximum execution time in seconds
- `max_retries`: Number of retry attempts on failure

### ExecutionMode Enum
Defines how steps should be executed.

#### Values
- `SERIAL`: Execute steps one after another (default)
- `CONCURRENT`: Execute steps simultaneously
- `CONDITIONAL`: Execute only if conditions are met

## Advanced Features

### Conditional Execution
```python
from tagent.pipeline.conditions import ConditionDSL

pipeline.step(
    "deep_analysis",
    "Perform deep analysis",
    depends_on=["initial_analysis"],
    condition=ConditionDSL.step_result_contains("initial_analysis", "requires_deep_dive")
)
```

### Memory Persistence
Pipeline steps can save and retrieve data across executions:

```python
# In a custom tool
def my_tool(state, args):
    # Save data for other steps
    pipeline_memory.save_step_result("my_step", {"key": "value"})
    
    # Retrieve data from previous steps
    previous_data = pipeline_memory.get_step_result("previous_step")
    
    return ("result", processed_data)
```

### Monitoring and Metrics
```python
from tagent.pipeline.monitoring import PipelineMonitor

monitor = PipelineMonitor()
monitor.start_monitoring("my_pipeline", total_steps=5)

# Monitor progress
progress = monitor.get_pipeline_progress("my_pipeline")
print(f"Progress: {progress.completion_percentage}%")
```

## Best Practices

### 1. Step Naming
- Use descriptive, unique names
- Follow consistent naming conventions
- Avoid spaces and special characters

### 2. Dependency Management
- Keep dependencies simple and clear
- Avoid circular dependencies
- Use meaningful dependency relationships

### 3. Error Handling
- Set appropriate timeouts for steps
- Configure retry logic for transient failures
- Use constraints to guide step behavior

### 4. Performance Optimization
- Use concurrent execution where appropriate
- Avoid unnecessarily complex dependency chains
- Monitor resource usage

## Troubleshooting

### Common Issues

#### Pipeline Validation Errors
```
PipelineValidationError: Step 'analyze' depends on unknown step 'search'
```
**Solution**: Ensure all step names in `depends_on` match existing step names.

#### Import Errors
```
ImportError: Pipeline support requires additional dependencies
```
**Solution**: Install pipeline dependencies with `pip install tagent[pipeline]`

#### Performance Issues
- Check for circular dependencies
- Reduce concurrent step limits
- Optimize step execution time

## Examples

### Company Research Pipeline
```python
company_pipeline = (Pipeline("company_research", "Research company information")
    .step("web_search", "Search for company information")
    .step("reviews", "Analyze customer reviews", depends_on=["web_search"])
    .step("social_media", "Check social media presence", 
          depends_on=["web_search"], 
          execution_mode=ExecutionMode.CONCURRENT)
    .step("report", "Generate final report", depends_on=["reviews", "social_media"])
)
```

### E-commerce Analysis Pipeline
```python
ecommerce_pipeline = (Pipeline("ecommerce_analysis", "E-commerce market analysis")
    .step("product_search", "Search for products")
    .step("competitor_analysis", "Analyze competitors", 
          depends_on=["product_search"],
          execution_mode=ExecutionMode.CONCURRENT)
    .step("market_trends", "Analyze market trends", 
          depends_on=["product_search"],
          execution_mode=ExecutionMode.CONCURRENT)
    .step("opportunities", "Identify opportunities", 
          depends_on=["competitor_analysis", "market_trends"])
)
```
```

### Acceptance Criteria
- [ ] All components have comprehensive unit tests
- [ ] Integration tests cover real-world scenarios
- [ ] Performance benchmarks establish baselines
- [ ] Documentation is complete and accurate
- [ ] Examples demonstrate key features

---

## Task 4.4: Migration and Compatibility (`src/tagent/migration/`)
**Priority: MEDIUM** | **Estimated Time: 3-4 days**

### Objective
Ensure smooth migration from single-goal to pipeline execution with full backward compatibility.

### Requirements
- [ ] Create migration utilities for existing code
- [ ] Add compatibility layer for legacy functions
- [ ] Implement version detection and warnings
- [ ] Create migration documentation
- [ ] Add automated migration tools

### Implementation Details

#### Migration Utilities
```python
# src/tagent/migration/pipeline_migration.py
class PipelineMigrator:
    """Utilities for migrating from single-goal to pipeline execution."""
    
    def __init__(self):
        self.conversion_patterns = {
            'simple_goal': self._convert_simple_goal,
            'multi_step_goal': self._convert_multi_step_goal,
            'conditional_goal': self._convert_conditional_goal
        }
    
    def detect_migration_opportunity(self, goal: str) -> Dict[str, Any]:
        """Detect if a goal could benefit from pipeline conversion."""
        analysis = {
            'should_migrate': False,
            'suggested_pattern': None,
            'benefits': [],
            'complexity': 'low'
        }
        
        # Analyze goal text for pipeline indicators
        if self._contains_multiple_steps(goal):
            analysis['should_migrate'] = True
            analysis['suggested_pattern'] = 'multi_step_goal'
            analysis['benefits'].append('Better step tracking')
            analysis['benefits'].append('Failure recovery')
            analysis['complexity'] = 'medium'
        
        if self._contains_conditional_logic(goal):
            analysis['should_migrate'] = True
            analysis['suggested_pattern'] = 'conditional_goal'
            analysis['benefits'].append('Conditional execution')
            analysis['complexity'] = 'high'
        
        return analysis
    
    def suggest_pipeline_conversion(self, goal: str) -> Pipeline:
        """Suggest a pipeline conversion for a goal."""
        # Analyze goal structure
        steps = self._extract_steps_from_goal(goal)
        
        # Create pipeline
        pipeline = Pipeline(
            name=self._generate_pipeline_name(goal),
            description=f"Auto-converted from goal: {goal[:100]}..."
        )
        
        # Add steps
        for i, step_text in enumerate(steps):
            step_name = f"step_{i+1}"
            dependencies = [f"step_{i}"] if i > 0 else []
            
            pipeline.step(
                name=step_name,
                goal=step_text,
                depends_on=dependencies
            )
        
        return pipeline
```

#### Compatibility Layer
```python
# src/tagent/compatibility.py
import warnings
from typing import Union, Optional, Dict, Any, Callable

def run_agent_legacy(
    goal: str,
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    max_iterations: int = 20,
    tools: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> Any:
    """
    Legacy run_agent function with deprecation warning.
    
    This function is deprecated. Use run_agent() with TAgentConfig instead.
    """
    warnings.warn(
        "run_agent_legacy is deprecated. Use run_agent() with TAgentConfig instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Convert to new format
    from .agent import run_agent
    from .config import TAgentConfig
    
    config = TAgentConfig(
        model=model,
        api_key=api_key,
        max_iterations=max_iterations,
        tools=tools or {},
        **kwargs
    )
    
    return run_agent(goal, config)

def detect_legacy_usage(func_name: str, args: tuple, kwargs: dict) -> bool:
    """Detect if function is being called with legacy parameters."""
    legacy_patterns = {
        'run_agent': ['model', 'api_key', 'max_iterations', 'tools'],
        'run_task_based_agent': ['model', 'api_key', 'max_iterations']
    }
    
    pattern = legacy_patterns.get(func_name, [])
    
    # Check if legacy parameters are being used
    for param in pattern:
        if param in kwargs:
            return True
    
    return False
```

### Acceptance Criteria
- [ ] Migration utilities work correctly
- [ ] Backward compatibility is maintained
- [ ] Deprecation warnings are appropriate
- [ ] Migration documentation is clear
- [ ] Automated tools are helpful

---

## Integration Requirements

### Final Integration Testing
- All phase 4 components work together
- Integration with previous phases is seamless
- Performance meets requirements
- Error handling is comprehensive

### Documentation Requirements
- Complete API documentation
- Migration guides
- Best practices
- Troubleshooting guides
- Performance considerations

### Deployment Requirements
- Package configuration is correct
- Installation works on all platforms
- Dependencies are properly managed
- Version compatibility is maintained

## Success Metrics
- All phase 4 tasks completed successfully
- Package can be installed and used as intended
- Documentation is comprehensive and accurate
- Migration path is clear and well-supported
- Performance meets established benchmarks

## Final Deliverables
1. Complete pipeline system implementation
2. Comprehensive test suite
3. Full documentation
4. Migration utilities
5. Performance benchmarks
6. Example templates
7. Package configuration
"""
Pipeline API for TAgent Pipeline System.

This module provides a high-level API for building, optimizing, and serializing
pipeline definitions.
"""

import json
import pickle
from typing import Any, Dict, List, Optional
import networkx as nx
import yaml

from .models import (
    Pipeline, PipelineStep, ExecutionMode, PipelineTemplate
)
from .conditions import AnyCondition


class PipelineValidationError(Exception):
    """Raised when pipeline validation fails."""
    pass


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
    
    def parallel_steps(self, *step_configs: Dict[str, Any]) -> 'PipelineBuilder':
        """Add multiple steps that can run in parallel."""
        for config in step_configs:
            config.setdefault('execution_mode', ExecutionMode.CONCURRENT)
            self.step(**config)
        return self
    
    def conditional_step(self, name: str, goal: str, condition: AnyCondition, **kwargs) -> 'PipelineBuilder':
        """Add a conditional step."""
        kwargs['condition'] = condition
        return self.step(name, goal, **kwargs)
    
    def add_global_constraint(self, constraint: str) -> 'PipelineBuilder':
        """Add a global constraint to the pipeline."""
        self.pipeline.add_constraint(constraint)
        return self
    
    def set_shared_data(self, key: str, value: Any) -> 'PipelineBuilder':
        """Set shared data for the pipeline."""
        if isinstance(value, str):
            self.pipeline.set_shared_variable(key, value)
        elif isinstance(value, (int, float)):
            self.pipeline.set_shared_number(key, value)
        elif isinstance(value, bool):
            self.pipeline.set_shared_flag(key, value)
        elif isinstance(value, list):
            self.pipeline.set_shared_list(key, value)
        else:
            raise TypeError(f"Unsupported shared data type: {type(value)}")
        return self
    
    def validate(self) -> 'PipelineBuilder':
        """Validate the pipeline configuration."""
        self.validation_errors = self.pipeline.validate()
        
        # Additional validation
        self._validate_step_dependencies()
        
        return self
    
    def build(self) -> Pipeline:
        """Build and return the final pipeline."""
        self.validate()
        
        if self.validation_errors:
            raise PipelineValidationError(
                f"Pipeline validation failed: {', '.join(self.validation_errors)}"
            )
        
        optimizer = PipelineOptimizer(self.pipeline)
        optimized_steps = optimizer.optimize_execution_order()
        self.pipeline.steps = optimized_steps
        
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


class PipelineOptimizer:
    """Optimizes pipeline execution for better performance."""
    
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.dependency_graph = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> nx.DiGraph:
        """Build a dependency graph."""
        graph = nx.DiGraph()
        for step in self.pipeline.steps:
            graph.add_node(step.name)
            for dep in step.depends_on:
                graph.add_edge(dep, step.name)
        return graph

    def optimize_execution_order(self) -> List[PipelineStep]:
        """Optimize step execution order for maximum parallelism."""
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            raise PipelineValidationError("Pipeline has a circular dependency.")

        # Topological sort gives a valid execution order
        # For more advanced optimization, we could group by levels
        optimized_order_names = list(nx.topological_sort(self.dependency_graph))
        
        step_map = {step.name: step for step in self.pipeline.steps}
        optimized_steps = [step_map[name] for name in optimized_order_names]
        
        return optimized_steps

    def identify_parallel_opportunities(self) -> List[List[str]]:
        """Identify steps that can be executed in parallel."""
        levels = []
        graph = self.dependency_graph.copy()
        
        while graph:
            zero_in_degree = [node for node, in_degree in graph.in_degree() if in_degree == 0]
            if not zero_in_degree:
                raise PipelineValidationError("Circular dependency detected in identify_parallel_opportunities.")
            
            levels.append(zero_in_degree)
            graph.remove_nodes_from(zero_in_degree)
            
        return levels


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
            return yaml.dump(pipeline_data, default_flow_style=False)
        elif format == 'pickle':
            return pickle.dumps(pipeline_data)
        return "" # Should not be reached
    
    def import_pipeline(self, data: str, format: str = 'json') -> Pipeline:
        """Import pipeline from specified format."""
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        pipeline_data: Dict[str, Any]
        if format == 'json':
            pipeline_data = json.loads(data)
        elif format == 'yaml':
            pipeline_data = yaml.safe_load(data)
        elif format == 'pickle':
            pipeline_data = pickle.loads(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return self._dict_to_pipeline(pipeline_data)
    
    def _pipeline_to_dict(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            'name': pipeline.name,
            'description': pipeline.description,
            'steps': [self._step_to_dict(step) for step in pipeline.steps],
            'global_constraints': pipeline.global_constraints,
            'shared_context': pipeline.shared_context.model_dump(),
            'version': '1.0'
        }
    
    def _step_to_dict(self, step: PipelineStep) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        # This is tricky because PipelineStep is a dataclass, not Pydantic model
        # and contains non-serializable types like AnyCondition
        # For now, I will serialize what I can.
        # A proper implementation might need to serialize conditions properly.
        return {
            'name': step.name,
            'goal': step.goal,
            'constraints': step.constraints,
            'depends_on': step.depends_on,
            'execution_mode': step.execution_mode.value,
            'timeout': step.timeout,
            'max_retries': step.max_retries,
            'tools_filter': step.tools_filter,
            'condition': str(step.condition) if step.condition else None, # Simplified serialization
        }

    def _dict_to_pipeline(self, data: Dict[str, Any]) -> Pipeline:
        """Convert dictionary to pipeline object."""
        pipeline = Pipeline(name=data['name'], description=data.get('description', ''))
        
        for step_data in data.get('steps', []):
            # Deserializing condition is complex and not supported in this simplified version.
            # The user would need to re-attach conditions manually after import.
            step_data.pop('condition', None)
            step_data['execution_mode'] = ExecutionMode(step_data['execution_mode'])
            pipeline.steps.append(PipelineStep(**step_data))
            
        pipeline.global_constraints = data.get('global_constraints', [])
        
        # shared_context needs to be handled carefully
        # Assuming it's a SharedPipelineContext model
        if 'shared_context' in data:
            # This part is complex as the shared_context in Pipeline is not a simple dict
            # I will skip this for now.
            pass

        return pipeline

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'model_dump_json'):
            return json.loads(obj.model_dump_json())
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


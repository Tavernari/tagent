"""
Pipeline Scheduler for TAgent Pipeline System.

This module implements dependency resolution and execution scheduling for pipeline steps.
It provides topological sorting, circular dependency detection, and step readiness management.
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque
from enum import Enum

from .models import Pipeline, PipelineStep, StepStatus


class SchedulingError(Exception):
    """Base exception for scheduling errors."""
    pass


class CircularDependencyError(SchedulingError):
    """Raised when circular dependencies are detected."""
    pass


class DependencyNotFoundError(SchedulingError):
    """Raised when a dependency step is not found."""
    pass


class SchedulingStrategy(Enum):
    """Strategies for scheduling pipeline execution."""
    TOPOLOGICAL = "topological"  # Standard topological sort
    PRIORITY = "priority"         # Priority-based scheduling
    RESOURCE_AWARE = "resource_aware"  # Resource-aware scheduling


class PipelineScheduler:
    """Manages pipeline execution order and dependencies."""
    
    def __init__(self, pipeline: Pipeline, strategy: SchedulingStrategy = SchedulingStrategy.TOPOLOGICAL):
        self.pipeline = pipeline
        self.strategy = strategy
        self.step_status: Dict[str, StepStatus] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.reverse_dependency_graph: Dict[str, List[str]] = {}
        self.execution_order: List[str] = []
        
        # Initialize step status
        for step in pipeline.steps:
            self.step_status[step.name] = step.status
        
        # Build dependency graphs
        self._build_dependency_graphs()
        
        # Validate dependencies
        self._validate_dependencies()
        
        # Generate execution order
        self._generate_execution_order()
    
    def _build_dependency_graphs(self):
        """Build dependency graphs using step names."""
        # Forward dependency graph (step -> dependencies)
        self.dependency_graph = {}
        
        # Reverse dependency graph (step -> dependents)
        self.reverse_dependency_graph = defaultdict(list)
        
        for step in self.pipeline.steps:
            self.dependency_graph[step.name] = step.depends_on.copy()
            
            # Build reverse graph
            for dependency in step.depends_on:
                self.reverse_dependency_graph[dependency].append(step.name)
        
        # Ensure all steps are in the reverse graph
        for step in self.pipeline.steps:
            if step.name not in self.reverse_dependency_graph:
                self.reverse_dependency_graph[step.name] = []
    
    def _validate_dependencies(self):
        """Validate dependency graph for missing dependencies and cycles."""
        step_names = {step.name for step in self.pipeline.steps}
        
        # Check for missing dependencies
        for step in self.pipeline.steps:
            for dependency in step.depends_on:
                if dependency not in step_names:
                    raise DependencyNotFoundError(
                        f"Step '{step.name}' depends on unknown step '{dependency}'"
                    )
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            cycle = self._find_circular_dependency()
            raise CircularDependencyError(
                f"Circular dependency detected: {' -> '.join(cycle)}"
            )
    
    def _has_circular_dependencies(self) -> bool:
        """Check if dependency graph has circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(step_name: str) -> bool:
            visited.add(step_name)
            rec_stack.add(step_name)
            
            for dependency in self.dependency_graph.get(step_name, []):
                if dependency not in visited:
                    if dfs(dependency):
                        return True
                elif dependency in rec_stack:
                    return True
            
            rec_stack.remove(step_name)
            return False
        
        for step_name in self.dependency_graph:
            if step_name not in visited:
                if dfs(step_name):
                    return True
        
        return False
    
    def _find_circular_dependency(self) -> List[str]:
        """Find and return one circular dependency cycle."""
        visited = set()
        rec_stack = set()
        parent = {}
        
        def dfs(step_name: str) -> Optional[List[str]]:
            visited.add(step_name)
            rec_stack.add(step_name)
            
            for dependency in self.dependency_graph.get(step_name, []):
                if dependency not in visited:
                    parent[dependency] = step_name
                    cycle = dfs(dependency)
                    if cycle:
                        return cycle
                elif dependency in rec_stack:
                    # Found cycle - build cycle path
                    cycle = [dependency]
                    current = step_name
                    while current != dependency:
                        cycle.append(current)
                        current = parent.get(current)
                    cycle.append(dependency)
                    return cycle
            
            rec_stack.remove(step_name)
            return None
        
        for step_name in self.dependency_graph:
            if step_name not in visited:
                cycle = dfs(step_name)
                if cycle:
                    return cycle
        
        return []
    
    def _generate_execution_order(self):
        """Generate execution order based on scheduling strategy."""
        if self.strategy == SchedulingStrategy.TOPOLOGICAL:
            self.execution_order = self._topological_sort()
        elif self.strategy == SchedulingStrategy.PRIORITY:
            self.execution_order = self._priority_sort()
        elif self.strategy == SchedulingStrategy.RESOURCE_AWARE:
            self.execution_order = self._resource_aware_sort()
        else:
            raise ValueError(f"Unknown scheduling strategy: {self.strategy}")
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort using Kahn's algorithm."""
        # Calculate in-degree for each step
        in_degree = {}
        for step_name in self.dependency_graph:
            in_degree[step_name] = len(self.dependency_graph[step_name])
        
        # Queue for steps with no dependencies
        queue = deque()
        for step_name, degree in in_degree.items():
            if degree == 0:
                queue.append(step_name)
        
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Reduce in-degree for dependent steps
            for dependent in self.reverse_dependency_graph.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check if all steps were processed
        if len(result) != len(self.pipeline.steps):
            raise CircularDependencyError("Circular dependency detected during topological sort")
        
        return result
    
    def _priority_sort(self) -> List[str]:
        """Sort based on priority (for future implementation)."""
        # For now, fall back to topological sort
        # In the future, this could consider step priorities
        return self._topological_sort()
    
    def _resource_aware_sort(self) -> List[str]:
        """Sort based on resource requirements (for future implementation)."""
        # For now, fall back to topological sort
        # In the future, this could consider resource constraints
        return self._topological_sort()
    
    def get_execution_order(self) -> List[str]:
        """Get the complete execution order for the pipeline."""
        return self.execution_order.copy()
    
    def get_ready_steps(self) -> List[str]:
        """Get steps that are ready for execution based on current status."""
        ready_steps = []
        
        for step_name in self.execution_order:
            step = self.pipeline.get_step_by_name(step_name)
            if not step:
                continue
                
            current_status = self.step_status.get(step_name, StepStatus.PENDING)
            
            # Skip if not pending
            if current_status != StepStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            dependencies_ready = True
            for dependency in self.dependency_graph.get(step_name, []):
                dep_status = self.step_status.get(dependency, StepStatus.PENDING)
                if dep_status != StepStatus.COMPLETED:
                    dependencies_ready = False
                    break
            
            if dependencies_ready:
                ready_steps.append(step_name)
        
        return ready_steps
    
    def get_blocked_steps(self) -> List[Tuple[str, List[str]]]:
        """Get steps that are blocked and their blocking dependencies."""
        blocked_steps = []
        
        for step_name in self.execution_order:
            step = self.pipeline.get_step_by_name(step_name)
            if not step:
                continue
                
            current_status = self.step_status.get(step_name, StepStatus.PENDING)
            
            # Skip if not pending
            if current_status != StepStatus.PENDING:
                continue
            
            # Find blocking dependencies
            blocking_deps = []
            for dependency in self.dependency_graph.get(step_name, []):
                dep_status = self.step_status.get(dependency, StepStatus.PENDING)
                if dep_status != StepStatus.COMPLETED:
                    blocking_deps.append(dependency)
            
            if blocking_deps:
                blocked_steps.append((step_name, blocking_deps))
        
        return blocked_steps
    
    def update_step_status(self, step_name: str, status: StepStatus):
        """Update the status of a step."""
        if step_name not in self.step_status:
            raise ValueError(f"Step '{step_name}' not found in pipeline")
        
        self.step_status[step_name] = status
        
        # Also update in pipeline
        step = self.pipeline.get_step_by_name(step_name)
        if step:
            step.status = status
    
    def get_step_dependencies(self, step_name: str) -> List[str]:
        """Get dependencies for a specific step."""
        return self.dependency_graph.get(step_name, []).copy()
    
    def get_step_dependents(self, step_name: str) -> List[str]:
        """Get steps that depend on the given step."""
        return self.reverse_dependency_graph.get(step_name, []).copy()
    
    def get_dependency_chain(self, step_name: str) -> List[str]:
        """Get the complete dependency chain for a step."""
        visited = set()
        chain = []
        
        def dfs(current_step: str):
            if current_step in visited:
                return
            visited.add(current_step)
            
            for dependency in self.dependency_graph.get(current_step, []):
                dfs(dependency)
            
            chain.append(current_step)
        
        dfs(step_name)
        return chain
    
    def get_concurrent_groups(self) -> List[List[str]]:
        """Get groups of steps that can run concurrently."""
        groups = []
        remaining_steps = set(self.execution_order)
        processed = set()
        
        while remaining_steps:
            current_group = []
            
            # Find steps that can run concurrently
            for step_name in list(remaining_steps):
                # Check if all dependencies are processed
                dependencies_ready = True
                for dependency in self.dependency_graph.get(step_name, []):
                    if dependency not in processed:
                        dependencies_ready = False
                        break
                
                if dependencies_ready:
                    current_group.append(step_name)
            
            # Remove current group from remaining steps
            for step_name in current_group:
                remaining_steps.remove(step_name)
                processed.add(step_name)
            
            if current_group:
                groups.append(current_group)
            else:
                # This shouldn't happen with valid dependency graph
                break
        
        return groups
    
    def is_step_ready(self, step_name: str) -> bool:
        """Check if a specific step is ready for execution."""
        return step_name in self.get_ready_steps()
    
    def has_deadlock(self) -> bool:
        """Check if there's a deadlock in the current execution state."""
        ready_steps = self.get_ready_steps()
        
        # Check if there are pending steps but no ready steps
        pending_steps = [
            name for name, status in self.step_status.items()
            if status == StepStatus.PENDING
        ]
        
        return len(pending_steps) > 0 and len(ready_steps) == 0
    
    def get_scheduling_summary(self) -> Dict[str, any]:
        """Get comprehensive scheduling summary."""
        ready_steps = self.get_ready_steps()
        blocked_steps = self.get_blocked_steps()
        concurrent_groups = self.get_concurrent_groups()
        
        status_counts = {}
        for status in StepStatus:
            status_counts[status.value] = sum(
                1 for s in self.step_status.values() if s == status
            )
        
        return {
            "strategy": self.strategy.value,
            "total_steps": len(self.pipeline.steps),
            "execution_order": self.execution_order,
            "ready_steps": ready_steps,
            "blocked_steps": blocked_steps,
            "concurrent_groups": concurrent_groups,
            "status_counts": status_counts,
            "has_deadlock": self.has_deadlock(),
            "dependency_graph": self.dependency_graph,
            "reverse_dependency_graph": dict(self.reverse_dependency_graph)
        }
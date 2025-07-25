"""
Pipeline Executor for TAgent Pipeline System.

This module provides the main pipeline execution engine with async support,
memory persistence, concurrent step execution, and comprehensive error handling.
"""

import asyncio
import logging
import inspect
from functools import wraps
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from ..task_agent import run_task_based_agent, TaskBasedAgentResult
from ..ui import print_retro_banner, print_retro_status, print_retro_step
from ..config import TAgentConfig
from .models import (
    Pipeline, PipelineStep, PipelineResult, StepStatus, 
    PipelineExecutionProgress, SharedPipelineContext, ExecutionMetadata,
    DefaultStepOutput
)
from .state import PipelineStateMachine, PipelineMemory, PipelinePhase
from .scheduler import PipelineScheduler
from .persistence import PipelineMemoryManager, PersistenceConfig, StorageBackendType
from .conditions import ConditionEvaluator, BaseCondition
from .monitoring import PipelineMonitor
from .models import ExecutionMode


logger = logging.getLogger(__name__)


class PipelineExecutionError(Exception):
    """Base exception for pipeline execution errors."""
    pass


class PipelineDeadlockError(PipelineExecutionError):
    """Raised when pipeline execution is deadlocked."""
    pass


class PipelineTimeoutError(PipelineExecutionError):
    """Raised when pipeline execution times out."""
    pass


class PipelineValidationError(PipelineExecutionError):
    """Raised when pipeline validation fails."""
    pass


class ExecutionStrategy(Enum):
    """Strategies for pipeline execution."""
    SEQUENTIAL = "sequential"      # Execute steps one by one
    PARALLEL = "parallel"          # Execute all ready steps in parallel
    OPTIMIZED = "optimized"        # Smart scheduling based on dependencies and resources


@dataclass
class PipelineExecutorConfig:
    """Configuration for pipeline executor."""
    max_concurrent_steps: int = 5
    step_timeout: Optional[int] = 300  # 5 minutes default
    pipeline_timeout: Optional[int] = 3600  # 1 hour default
    execution_strategy: ExecutionStrategy = ExecutionStrategy.OPTIMIZED
    enable_communication: bool = True
    enable_persistence: bool = False
    persistence_config: Optional[PersistenceConfig] = None
    retry_failed_steps: bool = True
    max_pipeline_retries: int = 3
    deadlock_detection_interval: int = 30  # seconds
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default persistence config if not provided."""
        if self.persistence_config is None:
            self.persistence_config = PersistenceConfig(
                backend_type=StorageBackendType.FILE_JSON,
                base_path="./pipeline_memory",
                backup_enabled=True
            )


class PipelineExecutor:
    """Main pipeline execution engine with async support."""
    
    def __init__(self, pipeline: Pipeline, config: Union[TAgentConfig, Dict[str, Any]], executor_config: Optional[PipelineExecutorConfig] = None):
        self.pipeline = pipeline
        
        # Handle config conversion
        if isinstance(config, dict):
            self.agent_config = TAgentConfig.from_dict(config)
        else:
            self.agent_config = config
            
        self.executor_config = executor_config or PipelineExecutorConfig()
        
        # Initialize components
        self.memory_manager = None
        if self.executor_config.enable_persistence:
            self.memory_manager = PipelineMemoryManager(self.executor_config.persistence_config)
        
        self.state_machine = PipelineStateMachine(pipeline, self.memory_manager)
        self.scheduler = PipelineScheduler(pipeline)
        self.condition_evaluator = ConditionEvaluator()
        self.monitor = PipelineMonitor()
        
        # Execution control
        self.executor_pool = asyncio.Semaphore(self.executor_config.max_concurrent_steps)
        self.is_cancelled = False
        self.execution_start_time = None
        self.execution_stats = {
            "steps_executed": 0,
            "steps_failed": 0,
            "total_execution_time": 0.0,
            "avg_step_time": 0.0,
            "retry_count": 0
        }
        
        # Communication (will be implemented in Task 2.4)
        self.communicator = None
        if self.executor_config.enable_communication:
            # Placeholder - will be implemented in communication.py
            pass
    
    async def execute(self) -> PipelineResult:
        """Execute the complete pipeline with comprehensive error handling."""
        self.execution_start_time = datetime.now()
        
        print_retro_banner(f"PIPELINE: {self.pipeline.name}", "═", 70)
        print_retro_status("INIT", f"Starting pipeline with {len(self.pipeline.steps)} steps")
        
        self.monitor.start_monitoring(self.pipeline.pipeline_id, len(self.pipeline.steps))
        
        try:
            # Validate pipeline before execution
            await self._validate_pipeline()
            
            # Initialize or restore pipeline state
            await self._initialize_pipeline_state()
            
            # Main execution loop
            result = await self._execute_pipeline_loop()
            
            # Create final result
            final_result = await self._create_pipeline_result(result)
            
            # Cleanup resources
            await self._cleanup_resources()
            
            print_retro_status("COMPLETE", f"Pipeline finished - Success: {final_result.success}")
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            
            # Handle errors with persistence for recovery
            await self._handle_pipeline_error(e)
            
            # Create error result
            error_result = await self._create_error_result(e)
            
            print_retro_status("ERROR", f"Pipeline failed: {str(e)}")
            raise PipelineExecutionError(f"Pipeline execution failed: {e}") from e
    
    async def _validate_pipeline(self):
        """Validate pipeline configuration and dependencies."""
        print_retro_step(0, "VALIDATE", "Validating pipeline configuration")
        
        # Basic pipeline validation
        validation_errors = self.pipeline.validate()
        if validation_errors:
            raise PipelineValidationError(f"Pipeline validation failed: {validation_errors}")
        
        # Check for cycles in dependencies
        try:
            self.scheduler.get_execution_order()
        except Exception as e:
            raise PipelineValidationError(f"Dependency validation failed: {e}")
        
        # Validate step configurations
        for step in self.pipeline.steps:
            if step.timeout and step.timeout <= 0:
                raise PipelineValidationError(f"Invalid timeout for step '{step.name}': {step.timeout}")
            
            if step.max_retries < 0:
                raise PipelineValidationError(f"Invalid max_retries for step '{step.name}': {step.max_retries}")
        
        logger.info(f"Pipeline '{self.pipeline.name}' validation successful")
    
    async def _initialize_pipeline_state(self):
        """Initialize or restore pipeline state from persistence."""
        print_retro_step(0, "INIT", "Initializing pipeline state")
        
        # Try to restore state if persistence is enabled
        if self.memory_manager:
            restored = await self.state_machine.restore_state()
            if restored:
                print_retro_status("RESTORED", "Pipeline state restored from persistence")
                logger.info(f"Pipeline state restored for '{self.pipeline.pipeline_id}'")
            else:
                print_retro_status("NEW", "Starting fresh pipeline execution")
                logger.info(f"Starting new pipeline execution for '{self.pipeline.pipeline_id}'")
        
        # Set initial phase
        self.state_machine.set_phase(PipelinePhase.SCHEDULING)
        
        # Update shared context
        self.pipeline.shared_context.execution_phase = PipelinePhase.SCHEDULING.value
    
    async def _execute_pipeline_loop(self) -> bool:
        """Main pipeline execution loop with dependency resolution."""
        print_retro_step(0, "EXECUTE", "Starting pipeline execution loop")
        
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0
        
        while not self.state_machine.is_pipeline_complete() and not self.is_cancelled:
            iteration += 1
            
            if iteration > max_iterations:
                raise PipelineExecutionError(f"Pipeline exceeded maximum iterations ({max_iterations})")
            
            # Check for timeout
            if await self._check_pipeline_timeout():
                raise PipelineTimeoutError("Pipeline execution timeout")
            
            # Get ready steps
            ready_steps = self.state_machine.get_ready_steps()
            
            if not ready_steps:
                # Check for deadlock
                if await self._is_deadlocked():
                    raise PipelineDeadlockError("Pipeline execution deadlocked - no steps can proceed")
                
                # All steps processed
                break
            
            print_retro_status("EXECUTING", f"Executing {len(ready_steps)} ready steps")
            
            # Execute ready steps based on strategy
            await self._execute_step_batch(ready_steps)
            
            # Persist state after each batch
            if self.memory_manager:
                await self._persist_pipeline_state()
            
            # Brief pause to prevent CPU spinning
            await asyncio.sleep(0.1)
        
        # Check if pipeline completed successfully
        success = self.state_machine.is_pipeline_complete() and not self.state_machine.has_pipeline_failed()
        
        if success:
            self.state_machine.set_phase(PipelinePhase.COMPLETED)
        else:
            self.state_machine.set_phase(PipelinePhase.FAILED)
        
        return success
    
    async def _execute_step_batch(self, steps: List[PipelineStep]):
        """Execute a batch of steps with proper concurrency control."""
        if not steps:
            return
        
        # Separate steps by execution mode
        serial_steps = []
        concurrent_steps = []
        
        for step in steps:
            if hasattr(step, 'execution_mode') and step.execution_mode == ExecutionMode.CONCURRENT:
                concurrent_steps.append(step)
            else:
                serial_steps.append(step)
        
        # Execute serial steps first (one by one)
        for step in serial_steps:
            if self.is_cancelled:
                break
            await self._execute_single_step(step)
        
        # Execute concurrent steps in parallel
        if concurrent_steps and not self.is_cancelled:
            # Create tasks for concurrent execution
            concurrent_tasks = [
                asyncio.create_task(self._execute_single_step(step))
                for step in concurrent_steps
            ]
            
            # Wait for all concurrent steps to complete
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            # Handle any exceptions from concurrent execution
            await self._handle_concurrent_step_results(concurrent_steps, results)

    async def _build_condition_context(self) -> Dict[str, Any]:
        """Build context for condition evaluation."""
        context = {}
        
        # Add step objects to context
        for step in self.pipeline.steps:
            context[step.name] = {"step": step}
        
        # Add shared context
        context["shared"] = self.pipeline.shared_context
        
        # Add step results from pipeline memory
        for step_name, mem_entry in self.state_machine.pipeline_memory.step_results.items():
            if step_name not in context:
                context[step_name] = {}
            context[step_name]["result"] = mem_entry.data

        return context

    async def _should_execute_step(self, step: PipelineStep) -> bool:
        """Check if a step should be executed based on its condition."""
        if step.condition is None:
            return True

        if step.condition_mode == "pre":
            context = await self._build_condition_context()
            
            # Support new BaseCondition classes
            if isinstance(step.condition, BaseCondition):
                return step.condition.evaluate(context)
            
            # Fall back to legacy condition evaluator
            return self.condition_evaluator.evaluate(step.condition, context)

        return True  # Post-conditions are not handled yet
    
    async def _execute_single_step(self, step: PipelineStep):
        """Execute individual step with comprehensive error handling."""
        async with self.executor_pool:
            step_id = f"{self.pipeline.name}:{step.name}"
            start_time = datetime.now()

            # Check pre-execution condition
            if not await self._should_execute_step(step):
                print_retro_banner(f"STEP: {step.name}", "─", 50)
                print_retro_status("SKIPPED", f"Step '{step.name}' skipped due to condition")
                print_retro_banner(f"END: {step.name}", "─", 50)
                self.state_machine.skip_step_execution(step)
                self.monitor.update_step_progress(self.pipeline.pipeline_id, step.name, StepStatus.SKIPPED)
                return

            # Add clear step start boundary
            print_retro_banner(f"STEP: {step.name}", "─", 50)
            print_retro_step(1, "EXECUTE", f"Executing step: {step.name}")
            
            try:
                # Mark step as running
                self.state_machine.start_step_execution(step)
                self.monitor.update_step_progress(self.pipeline.pipeline_id, step.name, StepStatus.RUNNING)
                
                # Prepare step context
                context = self.state_machine.prepare_step_context(step)
                
                # Execute step using TAgent with reduced verbosity
                result = await self._execute_tagent_step(step, context)
                
                # Complete step execution
                self.state_machine.complete_step_execution(step, result)
                self.monitor.update_step_progress(self.pipeline.pipeline_id, step.name, StepStatus.COMPLETED, result)
                
                # Update statistics
                execution_time = (datetime.now() - start_time).total_seconds()
                self.execution_stats["steps_executed"] += 1
                self.execution_stats["total_execution_time"] += execution_time
                self.execution_stats["avg_step_time"] = (
                    self.execution_stats["total_execution_time"] / self.execution_stats["steps_executed"]
                )
                
                # Broadcast completion event (placeholder for communication)
                if self.communicator:
                    await self.communicator.broadcast_event(
                        "step_completed",
                        {
                            'pipeline_id': self.pipeline.pipeline_id,
                            'step_name': step.name,
                            'result': result,
                            'execution_time': execution_time,
                            'timestamp': datetime.now()
                        },
                        self.pipeline.pipeline_id
                    )
                
                # Add clear step completion boundary
                print_retro_status("SUCCESS", f"Step '{step.name}' completed in {execution_time:.2f}s")
                print_retro_banner(f"END: {step.name}", "─", 50)
                logger.info(f"Step '{step.name}' completed successfully")
                
            except Exception as e:
                # Handle step failure with retry logic
                execution_time = (datetime.now() - start_time).total_seconds()
                await self._handle_step_error(step, e, execution_time)
                
                # Add clear step error boundary
                print_retro_banner(f"END: {step.name}", "─", 50)
                
                # Don't re-raise for individual steps - let the pipeline continue
                logger.error(f"Step '{step.name}' failed: {e}")
    
    def _create_enhanced_tool(self, original_tool: Callable, read_data_values: Dict[str, Any]) -> Callable:
        """Create enhanced tool that automatically injects read_data values as parameters."""
        
        # Get original tool signature
        sig = inspect.signature(original_tool)
        params = list(sig.parameters.keys())
        
        # Create wrapper function
        @wraps(original_tool)
        def enhanced_tool(state: dict, args: dict) -> Any:
            # Start with the original args
            enhanced_args = dict(args)
            
            # Generic parameter injection for other tools
            for param_name in params:
                if param_name in ['state', 'args']:
                    continue  # Skip standard parameters
                        
                # Check if any read_data value should be injected for this parameter
                for data_key, data_value in read_data_values.items():
                    # Try to match parameter name to data key or extract attribute name
                    if param_name == data_key:
                        enhanced_args[param_name] = data_value
                        break
                    elif '.' in data_key and param_name == data_key.split('.')[-1]:
                        enhanced_args[param_name] = data_value
                        break
            
            # Call original tool with enhanced args
            return original_tool(state, enhanced_args)
        
        # Preserve original function name and metadata
        enhanced_tool.__name__ = original_tool.__name__
        enhanced_tool.__doc__ = original_tool.__doc__
        
        return enhanced_tool

    async def _execute_tagent_step(self, step: PipelineStep, context) -> Any:
        """Execute step using TAgent with context injection."""
        # Convert context to format expected by TAgent
        step_goal = step.goal
        
        # Add context constraints
        if step.constraints:
            constraints_text = "\n".join([f"- {constraint}" for constraint in step.constraints])
            step_goal += f"\n\nConstraints:\n{constraints_text}"
        
        # Add dependency context
        if context.dependency_results:
            
            results = context.dependency_results
            if step.read_data:
                final_results = {}
                for read_path in step.read_data:
                    step_name, _, attr_name = read_path.partition('.')
                    if step_name in results:
                        if attr_name:
                            final_results[step_name] = getattr(results[step_name], attr_name, None)
                        else:
                            final_results[step_name] = results[step_name]
                    elif step_name in self.pipeline.shared_context.shared_variables:
                        if attr_name:
                            final_results[step_name] = getattr(self.pipeline.shared_context.shared_variables[step_name], attr_name, None)
                        else:
                            final_results[step_name] = self.pipeline.shared_context.shared_variables[step_name]
                    else:
                        raise ValueError(f"Step '{step_name}' not found in dependency results or shared context")
                        
                results = final_results

            dep_context = "\n".join([
                f"- {dep_name}: {result}" 
                for dep_name, result in results.items()
            ])
            step_goal += f"\n\nAvailable data from dependencies:\n{dep_context}"
        
        # Prepare tools: combine global and step-specific tools
        global_tools = self.agent_config.tools or []
        step_tools = step.tools or []
        
        # Use a dictionary to handle overrides and remove duplicates (step-specific takes precedence)
        combined_tools_dict = {tool.__name__: tool for tool in global_tools}
        combined_tools_dict.update({tool.__name__: tool for tool in step_tools})
        
        combined_tools = list(combined_tools_dict.values())
        
        # Apply tool filtering if specified
        available_tools = combined_tools
        if step.tools_filter:
            available_tools = [
                tool for tool in combined_tools
                if tool.__name__ in step.tools_filter
            ]
        
        # Create enhanced tools with read_data injection
        if step.read_data and results:
            enhanced_tools = []
            for tool in available_tools:
                enhanced_tool = self._create_enhanced_tool(tool, results)
                enhanced_tools.append(enhanced_tool)
            available_tools = enhanced_tools
        
        # Create step-specific config with fallback to main config
        if step.agent_config:
            # Use step-specific config as base, with fallback to main config
            step_config = TAgentConfig(
                model=step.agent_config.model or self.agent_config.model,
                api_key=step.agent_config.api_key or self.agent_config.api_key,
                max_iterations=step.agent_config.max_iterations or self.agent_config.max_iterations,
                verbose=step.agent_config.verbose if step.agent_config.verbose is not None else False,  # Force verbose=False for cleaner pipeline output
                tools=available_tools,  # Always use filtered tools from pipeline
                output_format=step.output_schema or step.agent_config.output_format or self.agent_config.output_format,
                ui_style=step.agent_config.ui_style or self.agent_config.ui_style,
                temperature=step.agent_config.temperature if step.agent_config.temperature is not None else self.agent_config.temperature
            )
        else:
            # Use main config as fallback with reduced verbosity for cleaner pipeline output
            step_config = TAgentConfig(
                model=self.agent_config.model,
                api_key=self.agent_config.api_key,
                max_iterations=self.agent_config.max_iterations,
                verbose=False,  # Force verbose=False for cleaner pipeline output
                tools=available_tools,
                output_format=step.output_schema or self.agent_config.output_format,
                ui_style=self.agent_config.ui_style,
                temperature=self.agent_config.temperature
            )
        
        # Execute with timeout if specified
        if step.timeout:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_task_based_agent,
                        step_goal,
                        step_config.tools,
                        output_format=step_config.output_format,
                        model=step_config.model,
                        api_key=step_config.api_key,
                        max_iterations=step_config.max_iterations,
                        verbose=step_config.verbose,
                        temperature=step_config.temperature
                    ),
                    timeout=step.timeout
                )
            except asyncio.TimeoutError:
                raise PipelineTimeoutError(f"Step '{step.name}' timed out after {step.timeout} seconds")
        else:
            result = await asyncio.to_thread(
                run_task_based_agent,
                step_goal,
                step_config.tools,
                output_format=step_config.output_format,
                model=step_config.model,
                api_key=step_config.api_key,
                max_iterations=step_config.max_iterations,
                verbose=step_config.verbose,
                temperature=step_config.temperature
            )
        
        # Extract result from TAgent response
        if hasattr(result, 'output') and result.output is not None:
            return result.output
        else:
            # Return a default structure if no specific output
            return DefaultStepOutput(
                result=str(result),
                summary=f"Step '{step.name}' executed successfully",
                success=True
            )
    
    async def _handle_step_error(self, step: PipelineStep, error: Exception, execution_time: float):
        """Handle step execution error with retry logic."""
        error_message = str(error)
        
        # Try to retry the step
        if self.state_machine.fail_step_execution(step, error_message):
            # Step will be retried
            self.execution_stats["retry_count"] += 1
            print_retro_status("RETRY", f"Step '{step.name}' will be retried (attempt {step.retry_count + 1})")
            logger.warning(f"Step '{step.name}' failed, will retry: {error_message}")
        else:
            # Max retries reached
            self.execution_stats["steps_failed"] += 1
            self.monitor.update_step_progress(self.pipeline.pipeline_id, step.name, StepStatus.FAILED, error_message)
            print_retro_status("FAILED", f"Step '{step.name}' failed permanently: {error_message}")
            logger.error(f"Step '{step.name}' failed permanently: {error_message}")
        
        # Broadcast error event (placeholder for communication)
        if self.communicator:
            await self.communicator.broadcast_event(
                "step_failed",
                {
                    'pipeline_id': self.pipeline.pipeline_id,
                    'step_name': step.name,
                    'error': error_message,
                    'execution_time': execution_time,
                    'retry_count': step.retry_count,
                    'timestamp': datetime.now()
                },
                self.pipeline.pipeline_id
            )
    
    async def _handle_concurrent_step_results(self, steps: List[PipelineStep], results: List[Any]):
        """Handle results from concurrent step execution."""
        for step, result in zip(steps, results):
            if isinstance(result, Exception):
                await self._handle_step_error(step, result, 0.0)
            # Success cases are already handled in _execute_single_step
    
    async def _is_deadlocked(self) -> bool:
        """Check if pipeline execution is deadlocked."""
        # Use scheduler's deadlock detection
        return self.scheduler.has_deadlock()
    
    async def _check_pipeline_timeout(self) -> bool:
        """Check if pipeline has exceeded timeout."""
        if not self.executor_config.pipeline_timeout:
            return False
        
        if not self.execution_start_time:
            return False
        
        elapsed = (datetime.now() - self.execution_start_time).total_seconds()
        return elapsed > self.executor_config.pipeline_timeout
    
    async def _persist_pipeline_state(self):
        """Persist current pipeline state."""
        if self.memory_manager:
            try:
                await self.memory_manager.persist_memory(
                    self.pipeline.pipeline_id,
                    self.state_machine.pipeline_memory
                )
            except Exception as e:
                logger.warning(f"Failed to persist pipeline state: {e}")
    
    async def _create_pipeline_result(self, success: bool) -> PipelineResult:
        """Create final pipeline result."""
        end_time = datetime.now()
        execution_time = (end_time - self.execution_start_time).total_seconds()
        
        result = self.state_machine.create_pipeline_result(success)
        result.execution_time = execution_time
        result.end_time = end_time
        
        # Add execution statistics
        result.retry_count = self.execution_stats["retry_count"]
        
        return result
    
    async def _create_error_result(self, error: Exception) -> PipelineResult:
        """Create result for failed pipeline execution."""
        end_time = datetime.now()
        execution_time = (end_time - self.execution_start_time).total_seconds()
        
        result = PipelineResult(
            pipeline_name=self.pipeline.name,
            pipeline_id=self.pipeline.pipeline_id,
            success=False,
            execution_time=execution_time,
            start_time=self.execution_start_time,
            end_time=end_time,
            total_steps=len(self.pipeline.steps),
            steps_completed=self.execution_stats["steps_executed"],
            steps_failed=self.execution_stats["steps_failed"],
            retry_count=self.execution_stats["retry_count"]
        )
        
        # Add error details
        result.add_step_error("pipeline_execution", str(error))
        
        return result
    
    async def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline-level errors with persistence."""
        logger.error(f"Pipeline error: {error}")
        
        # Persist error state for recovery
        if self.memory_manager:
            try:
                await self._persist_pipeline_state()
            except Exception as persist_error:
                logger.error(f"Failed to persist error state: {persist_error}")
    
    async def _cleanup_resources(self):
        """Cleanup resources after pipeline execution."""
        # Close executor pool
        # Note: Semaphore doesn't need explicit cleanup
        
        # Cleanup memory manager if needed
        if self.memory_manager:
            # Any cleanup needed for memory manager
            pass
        
        # Set cancellation flag
        self.is_cancelled = False
        
        logger.info(f"Pipeline '{self.pipeline.name}' resources cleaned up")
    
    def cancel(self):
        """Cancel pipeline execution."""
        self.is_cancelled = True
        logger.info(f"Pipeline '{self.pipeline.name}' execution cancelled")
    
    def get_execution_progress(self) -> PipelineExecutionProgress:
        """Get current execution progress."""
        return self.state_machine.get_execution_progress()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self.execution_stats,
            "pipeline_id": self.pipeline.pipeline_id,
            "pipeline_name": self.pipeline.name,
            "current_phase": self.state_machine.current_phase.value if self.state_machine.current_phase else None,
            "is_cancelled": self.is_cancelled,
            "execution_start_time": self.execution_start_time.isoformat() if self.execution_start_time else None
        }
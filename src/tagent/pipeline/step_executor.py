"""
Step Execution Wrapper for TAgent Pipeline System.

This module provides a sophisticated wrapper around the existing run_task_based_agent
function to integrate seamlessly with pipeline execution, including context injection,
timeout handling, retry logic, and tool filtering.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import random

from ..task_agent import run_task_based_agent, TaskBasedAgentResult
from ..config import TAgentConfig
from .models import (
    PipelineStep, PipelineStepContext, DefaultStepOutput,
    ExecutionMetadata, StepExecutionSummary
)
from .state import PipelineMemory


logger = logging.getLogger(__name__)


class StepExecutionError(Exception):
    """Base exception for step execution errors."""
    pass


class StepTimeoutError(StepExecutionError):
    """Raised when step execution times out."""
    pass


class StepValidationError(StepExecutionError):
    """Raised when step validation fails."""
    pass


class RetryableError(StepExecutionError):
    """Base class for errors that should trigger retries."""
    pass


class TemporaryServiceError(RetryableError):
    """Temporary service error that should be retried."""
    pass


class APIRateLimitError(RetryableError):
    """API rate limit error that should be retried."""
    pass


class RetryStrategy(Enum):
    """Strategies for retry backoff."""
    FIXED = "fixed"                # Fixed delay
    LINEAR = "linear"              # Linear backoff
    EXPONENTIAL = "exponential"    # Exponential backoff
    JITTERED = "jittered"         # Exponential backoff with jitter


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0        # Base delay in seconds
    max_delay: float = 60.0        # Maximum delay in seconds
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1      # Jitter as fraction of delay
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        APIRateLimitError,
        TemporaryServiceError
    )


@dataclass
class TimeoutConfig:
    """Configuration for timeout handling."""
    default_timeout: Optional[int] = 300  # 5 minutes
    warning_threshold: float = 0.8        # Warn at 80% of timeout
    grace_period: int = 10                # Grace period for cleanup


class TimeoutManager:
    """Manages step execution timeouts with warnings and graceful cleanup."""
    
    def __init__(self, config: TimeoutConfig = None):
        self.config = config or TimeoutConfig()
        self.active_timeouts: Dict[str, asyncio.Task] = {}
    
    async def execute_with_timeout(
        self, 
        coro, 
        timeout: Optional[int], 
        step_name: str = "unknown"
    ) -> Any:
        """Execute coroutine with timeout and warning system."""
        effective_timeout = timeout or self.config.default_timeout
        
        if effective_timeout is None:
            return await coro
        
        warning_time = effective_timeout * self.config.warning_threshold
        
        try:
            # Start warning task
            warning_task = asyncio.create_task(
                self._timeout_warning(warning_time, step_name)
            )
            self.active_timeouts[step_name] = warning_task
            
            # Execute with timeout
            result = await asyncio.wait_for(coro, timeout=effective_timeout)
            
            # Cancel warning if completed early
            if not warning_task.done():
                warning_task.cancel()
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Step '{step_name}' timed out after {effective_timeout}s")
            raise StepTimeoutError(f"Step '{step_name}' timed out after {effective_timeout} seconds")
        
        finally:
            # Cleanup
            if step_name in self.active_timeouts:
                task = self.active_timeouts[step_name]
                if not task.done():
                    task.cancel()
                del self.active_timeouts[step_name]
    
    async def _timeout_warning(self, warning_delay: float, step_name: str):
        """Issue warning when step is approaching timeout."""
        try:
            await asyncio.sleep(warning_delay)
            logger.warning(f"Step '{step_name}' is approaching timeout threshold")
        except asyncio.CancelledError:
            # Normal cancellation when step completes early
            pass


class RetryManager:
    """Manages retry logic with configurable backoff strategies."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    async def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt based on strategy."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        
        elif self.config.strategy == RetryStrategy.JITTERED:
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
            jitter = base_delay * self.config.jitter_range * (random.random() - 0.5)
            delay = base_delay + jitter
        
        else:
            delay = self.config.base_delay
        
        # Cap at maximum delay
        return min(delay, self.config.max_delay)
    
    def is_retryable_error(self, error: Exception) -> bool:
        """Determine if error should trigger retry."""
        return isinstance(error, self.config.retryable_exceptions)
    
    async def should_retry(self, error: Exception, attempt: int, max_retries: int) -> bool:
        """Determine if step should be retried."""
        if attempt >= max_retries:
            return False
        
        return self.is_retryable_error(error)


class ToolFilter:
    """Manages tool filtering for pipeline steps."""
    
    @staticmethod
    def filter_tools(
        available_tools: Dict[str, Callable], 
        tools_filter: Optional[List[str]]
    ) -> Dict[str, Callable]:
        """Filter tools based on step configuration."""
        if not tools_filter:
            return available_tools
        
        # Include only specified tools
        filtered_tools = {}
        for tool_name in tools_filter:
            if tool_name in available_tools:
                filtered_tools[tool_name] = available_tools[tool_name]
            else:
                logger.warning(f"Tool '{tool_name}' not found in available tools")
        
        return filtered_tools
    
    @staticmethod
    def validate_tools(tools: Dict[str, Callable]) -> List[str]:
        """Validate tool configurations and return list of issues."""
        issues = []
        
        for tool_name, tool_func in tools.items():
            if not callable(tool_func):
                issues.append(f"Tool '{tool_name}' is not callable")
            
            # Additional validation can be added here
        
        return issues


class ContextInjector:
    """Manages context injection for pipeline steps."""
    
    @staticmethod
    def prepare_step_context(
        step: PipelineStep,
        context: PipelineStepContext,
        pipeline_memory: PipelineMemory
    ) -> Dict[str, Any]:
        """Prepare comprehensive execution context with memory injection."""
        step_context = {}
        
        # Basic step information
        step_context.update({
            'step_name': step.name,
            'step_goal': step.goal,
            'pipeline_id': context.pipeline_id,
            'pipeline_name': context.pipeline_name,
            'execution_attempt': context.retry_count,
            'max_retries': context.max_retries,
            'constraints': step.constraints
        })
        
        # Inject dependency results with clear naming
        dependency_context = {}
        for dep_name, result in context.dependency_results.items():
            # Make dependency results easily accessible
            dependency_context[f"dependency_{dep_name}"] = result
            dependency_context[f"{dep_name}_result"] = result
        
        step_context['dependencies'] = dependency_context
        step_context.update(dependency_context)
        
        # Add shared context variables
        shared_ctx = context.shared_context
        step_context.update({
            'shared_variables': shared_ctx.shared_variables,
            'shared_numbers': shared_ctx.shared_numbers,
            'shared_flags': shared_ctx.shared_flags,
            'shared_lists': shared_ctx.shared_lists,
            'global_constraints': shared_ctx.global_constraints
        })
        
        # Add execution metadata
        metadata = context.execution_metadata
        step_context.update({
            'current_phase': metadata.current_phase,
            'execution_time': metadata.execution_time,
            'timeout': metadata.timeout,
            'tools_filter': metadata.tools_filter
        })
        
        # Add pipeline-wide memory data (if any additional shared data exists)
        memory_context = pipeline_memory.get_step_context(step.name)
        step_context['memory_context'] = memory_context.dict()
        
        return step_context
    
    @staticmethod
    def build_enhanced_goal(step: PipelineStep, context: Dict[str, Any]) -> str:
        """Build enhanced goal with context injection."""
        enhanced_goal = step.goal
        
        # Add constraints section
        if step.constraints:
            constraints_text = "\n".join([f"- {constraint}" for constraint in step.constraints])
            enhanced_goal += f"\n\nConstraints:\n{constraints_text}"
        
        # Add dependency context
        if 'dependencies' in context and context['dependencies']:
            dep_text = "\n".join([
                f"- {dep_name}: {result}" 
                for dep_name, result in context['dependencies'].items()
                if not dep_name.startswith('dependency_')  # Avoid duplicates
            ])
            enhanced_goal += f"\n\nAvailable data from previous steps:\n{dep_text}"
        
        # Add shared context if relevant
        shared_vars = context.get('shared_variables', {})
        if shared_vars:
            vars_text = "\n".join([
                f"- {var_name}: {value}" 
                for var_name, value in shared_vars.items()
            ])
            enhanced_goal += f"\n\nShared pipeline variables:\n{vars_text}"
        
        # Add global constraints
        global_constraints = context.get('global_constraints', [])
        if global_constraints:
            global_text = "\n".join([f"- {constraint}" for constraint in global_constraints])
            enhanced_goal += f"\n\nGlobal pipeline constraints:\n{global_text}"
        
        return enhanced_goal


class StepExecutor:
    """Sophisticated wrapper for executing individual pipeline steps."""
    
    def __init__(
        self, 
        agent_config: TAgentConfig,
        timeout_config: Optional[TimeoutConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        self.agent_config = agent_config
        self.timeout_manager = TimeoutManager(timeout_config)
        self.retry_manager = RetryManager(retry_config)
        self.tool_filter = ToolFilter()
        self.context_injector = ContextInjector()
        
        # Execution statistics
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retries_performed": 0,
            "timeouts_occurred": 0,
            "avg_execution_time": 0.0
        }
    
    async def execute_step(
        self,
        step: PipelineStep,
        context: PipelineStepContext,
        pipeline_memory: PipelineMemory
    ) -> Any:
        """Execute a single pipeline step with comprehensive error handling."""
        step_start_time = datetime.now()
        
        try:
            # Validate step configuration
            self._validate_step(step)
            
            # Prepare execution context
            step_context = self.context_injector.prepare_step_context(
                step, context, pipeline_memory
            )
            
            # Apply tool filtering
            available_tools = self.tool_filter.filter_tools(
                self.agent_config.tools or {}, 
                step.tools_filter
            )
            
            # Execute with retry logic
            result = await self._execute_with_retry(step, step_context, available_tools)
            
            # Update statistics
            execution_time = (datetime.now() - step_start_time).total_seconds()
            self._update_success_stats(execution_time)
            
            return result
            
        except Exception as e:
            # Update failure statistics
            execution_time = (datetime.now() - step_start_time).total_seconds()
            self._update_failure_stats(execution_time)
            
            logger.error(f"Step '{step.name}' execution failed: {e}")
            raise
    
    async def _execute_with_retry(
        self,
        step: PipelineStep,
        step_context: Dict[str, Any],
        available_tools: Dict[str, Callable]
    ) -> Any:
        """Execute step with retry logic and timeout handling."""
        last_exception = None
        
        for attempt in range(step.max_retries + 1):
            try:
                # Update attempt in context
                step_context['execution_attempt'] = attempt
                
                # Execute step with timeout
                result = await self._execute_with_timeout(
                    step, step_context, available_tools
                )
                
                # Validate and format result
                formatted_result = self._format_step_result(step, result)
                
                # Log successful execution
                if attempt > 0:
                    logger.info(f"Step '{step.name}' succeeded on attempt {attempt + 1}")
                
                return formatted_result
                
            except StepTimeoutError:
                last_exception = StepTimeoutError(
                    f"Step '{step.name}' timed out after {step.timeout} seconds"
                )
                self.execution_stats["timeouts_occurred"] += 1
                
                if attempt < step.max_retries:
                    await self._handle_timeout_retry(step, attempt)
                else:
                    raise last_exception
                    
            except Exception as e:
                last_exception = e
                
                if (attempt < step.max_retries and 
                    await self.retry_manager.should_retry(e, attempt, step.max_retries)):
                    
                    await self._handle_error_retry(step, e, attempt)
                else:
                    raise StepExecutionError(f"Step '{step.name}' failed: {str(e)}") from e
        
        # Should not reach here, but just in case
        raise last_exception or StepExecutionError(f"Step '{step.name}' failed after all retries")
    
    async def _execute_with_timeout(
        self,
        step: PipelineStep,
        step_context: Dict[str, Any],
        available_tools: Dict[str, Callable]
    ) -> TaskBasedAgentResult:
        """Execute step with timeout protection."""
        # Build enhanced goal with context
        enhanced_goal = self.context_injector.build_enhanced_goal(step, step_context)
        
        # Create step-specific config
        step_config = TAgentConfig(
            model=self.agent_config.model,
            api_key=self.agent_config.api_key,
            max_iterations=self.agent_config.max_iterations,
            verbose=self.agent_config.verbose,
            tools=available_tools,
            output_format=step.output_schema or self.agent_config.output_format,
            ui_style=self.agent_config.ui_style,
            temperature=self.agent_config.temperature
        )
        
        # Execute with timeout
        execution_coro = asyncio.to_thread(
            run_task_based_agent,
            enhanced_goal,
            config=step_config
        )
        
        result = await self.timeout_manager.execute_with_timeout(
            execution_coro,
            step.timeout,
            step.name
        )
        
        return result
    
    async def _handle_timeout_retry(self, step: PipelineStep, attempt: int):
        """Handle timeout retry with backoff."""
        delay = await self.retry_manager.calculate_delay(attempt)
        
        logger.warning(
            f"Step '{step.name}' timed out (attempt {attempt + 1}), "
            f"retrying in {delay:.2f}s"
        )
        
        self.execution_stats["retries_performed"] += 1
        await asyncio.sleep(delay)
    
    async def _handle_error_retry(self, step: PipelineStep, error: Exception, attempt: int):
        """Handle error retry with exponential backoff."""
        delay = await self.retry_manager.calculate_delay(attempt)
        
        logger.warning(
            f"Step '{step.name}' failed with {type(error).__name__} "
            f"(attempt {attempt + 1}), retrying in {delay:.2f}s: {str(error)}"
        )
        
        self.execution_stats["retries_performed"] += 1
        await asyncio.sleep(delay)
    
    def _validate_step(self, step: PipelineStep):
        """Validate step configuration before execution."""
        if not step.name:
            raise StepValidationError("Step name cannot be empty")
        
        if not step.goal:
            raise StepValidationError("Step goal cannot be empty")
        
        if step.timeout is not None and step.timeout <= 0:
            raise StepValidationError(f"Invalid timeout for step '{step.name}': {step.timeout}")
        
        if step.max_retries < 0:
            raise StepValidationError(f"Invalid max_retries for step '{step.name}': {step.max_retries}")
    
    def _format_step_result(self, step: PipelineStep, result: TaskBasedAgentResult) -> Any:
        """Format and validate step result."""
        # Extract output from TAgent result
        if hasattr(result, 'output') and result.output is not None:
            output = result.output
        else:
            # Create default output structure
            output = DefaultStepOutput(
                result=str(result),
                summary=f"Step '{step.name}' executed successfully",
                success=getattr(result, 'goal_achieved', True)
            )
        
        # Validate against output schema if specified
        if step.output_schema:
            try:
                validated_output = step.validate_output(output)
                return validated_output
            except Exception as e:
                raise StepValidationError(
                    f"Step '{step.name}' output validation failed: {e}"
                ) from e
        
        return output
    
    def _update_success_stats(self, execution_time: float):
        """Update statistics for successful execution."""
        self.execution_stats["total_executions"] += 1
        self.execution_stats["successful_executions"] += 1
        
        # Update average execution time
        total_time = (
            self.execution_stats["avg_execution_time"] * 
            (self.execution_stats["total_executions"] - 1) + execution_time
        )
        self.execution_stats["avg_execution_time"] = total_time / self.execution_stats["total_executions"]
    
    def _update_failure_stats(self, execution_time: float):
        """Update statistics for failed execution."""
        self.execution_stats["total_executions"] += 1
        self.execution_stats["failed_executions"] += 1
        
        # Update average execution time
        total_time = (
            self.execution_stats["avg_execution_time"] * 
            (self.execution_stats["total_executions"] - 1) + execution_time
        )
        self.execution_stats["avg_execution_time"] = total_time / self.execution_stats["total_executions"]
    
    def get_execution_summary(self) -> StepExecutionSummary:
        """Get execution statistics summary."""
        return StepExecutionSummary(
            step_name="executor_summary",
            status="running",
            execution_time=self.execution_stats["avg_execution_time"],
            retry_count=self.execution_stats["retries_performed"],
            error_message=None,
            result_available=True
        )
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed execution statistics."""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_executions"], 1)
            ) * 100,
            "failure_rate": (
                self.execution_stats["failed_executions"] / 
                max(self.execution_stats["total_executions"], 1)
            ) * 100,
            "retry_rate": (
                self.execution_stats["retries_performed"] / 
                max(self.execution_stats["total_executions"], 1)
            ),
            "timeout_rate": (
                self.execution_stats["timeouts_occurred"] / 
                max(self.execution_stats["total_executions"], 1)
            ) * 100
        }
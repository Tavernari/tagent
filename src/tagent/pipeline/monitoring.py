"""
Pipeline Monitoring for TAgent Pipeline System.

This module provides comprehensive monitoring for pipeline execution, including
real-time progress tracking, performance metrics collection, and failure analysis.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import psutil

from .models import (
    PipelineProgress, PipelineMetrics, FailureReport, StepStatus
)
from .persistence import StorageBackend


logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alerts for pipeline events (placeholder)."""
    def __init__(self):
        pass

    def check_alerts(self, pipeline_id: str, progress: PipelineProgress):
        # Placeholder for alert logic
        pass


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
        status: StepStatus,
        result: Any = None
    ):
        """Update progress for a specific step."""
        if pipeline_id not in self.active_pipelines:
            return

        progress = self.active_pipelines[pipeline_id]
        progress.update_step(step_name, status.value)

        self.metrics_collector.record_step_event(
            pipeline_id,
            step_name,
            status,
            datetime.now()
        )
        self.alert_manager.check_alerts(pipeline_id, progress)
        self._notify_progress_callbacks(pipeline_id, progress)

    def get_pipeline_progress(self, pipeline_id: str) -> Optional[PipelineProgress]:
        """Get current progress for a pipeline."""
        return self.active_pipelines.get(pipeline_id)

    def get_all_active_pipelines(self) -> Dict[str, PipelineProgress]:
        """Get progress for all active pipelines."""
        return self.active_pipelines.copy()

    def _notify_progress_callbacks(self, pipeline_id: str, progress: PipelineProgress):
        for callback in self.progress_callbacks:
            try:
                callback(pipeline_id, progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")


class MetricsCollector:
    """Collects and aggregates pipeline performance metrics."""

    def __init__(self):
        self.metrics_storage: Dict[str, Any] = {}
        self.real_time_metrics: Dict[str, Any] = {}

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
        status: StepStatus,
        timestamp: datetime
    ):
        """Record a step event with metrics."""
        if pipeline_id not in self.real_time_metrics:
            return

        metrics = self.real_time_metrics[pipeline_id]
        
        if status == StepStatus.RUNNING:
            metrics['step_durations'][step_name] = {
                'start_time': timestamp,
                'duration': None
            }
        elif status == StepStatus.COMPLETED:
            if step_name in metrics['step_durations']:
                start_time = metrics['step_durations'][step_name]['start_time']
                duration = (timestamp - start_time).total_seconds()
                metrics['step_durations'][step_name]['duration'] = duration
        elif status == StepStatus.FAILED:
            metrics['error_counts'] += 1

    def record_resource_usage(self, pipeline_id: str):
        """Record current resource usage."""
        if pipeline_id not in self.real_time_metrics:
            return
        
        usage = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024)
        }
        self.real_time_metrics[pipeline_id]['resource_usage'].append(usage)

    def get_pipeline_metrics(self, pipeline_id: str) -> Optional[PipelineMetrics]:
        """Get aggregated metrics for a pipeline."""
        if pipeline_id not in self.real_time_metrics:
            return None

        metrics = self.real_time_metrics[pipeline_id]
        total_duration = time.time() - metrics['start_time']
        step_durations = [
            d['duration'] for d in metrics['step_durations'].values()
            if d['duration'] is not None
        ]

        return PipelineMetrics(
            pipeline_id=pipeline_id,
            total_duration=total_duration,
            average_step_duration=sum(step_durations) / len(step_durations) if step_durations else 0,
            max_step_duration=max(step_durations) if step_durations else 0,
            min_step_duration=min(step_durations) if step_durations else 0,
            error_count=metrics['error_counts'],
            steps_completed=len(step_durations),
            resource_usage_summary=self._summarize_resource_usage(metrics['resource_usage'])
        )

    def _summarize_resource_usage(self, usage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not usage_data:
            return {}
        
        avg_cpu = sum(d['cpu_percent'] for d in usage_data) / len(usage_data)
        avg_mem = sum(d['memory_percent'] for d in usage_data) / len(usage_data)
        max_mem = max(d['memory_used_mb'] for d in usage_data)
        
        return {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_mem,
            'max_memory_used_mb': max_mem,
        }


class FailureAnalyzer:
    """Analyzes pipeline failures and generates reports."""

    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend

    async def analyze_failure(
        self,
        pipeline_id: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> FailureReport:
        """Analyze a pipeline failure and generate report."""
        failure_report = FailureReport(
            pipeline_id=pipeline_id,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            analysis=self._analyze_error_pattern(error),
            recommendations=self._generate_recommendations(error, context)
        )
        
        await self.storage.save(
            f"failure_report_{pipeline_id}_{int(time.time())}",
            failure_report.model_dump()
        )
        return failure_report

    def _analyze_error_pattern(self, error: Exception) -> Dict[str, Any]:
        """Analyze error patterns for insights."""
        return {
            'error_category': self._categorize_error(error),
            'is_retryable': self._is_retryable_error(error),
            'potential_causes': self._identify_potential_causes(error),
        }

    def _categorize_error(self, error: Exception) -> str:
        if isinstance(error, ConnectionError):
            return "Network"
        elif isinstance(error, TimeoutError):
            return "Timeout"
        elif isinstance(error, MemoryError):
            return "Resource"
        elif isinstance(error, (ValueError, TypeError)):
            return "Data"
        else:
            return "Unknown"

    def _is_retryable_error(self, error: Exception) -> bool:
        return isinstance(error, (ConnectionError, TimeoutError))

    def _identify_potential_causes(self, error: Exception) -> List[str]:
        if isinstance(error, ConnectionError):
            return ["Network issue", "API endpoint down", "Firewall issue"]
        elif isinstance(error, TimeoutError):
            return ["Slow network", "LLM API slow", "Complex task"]
        return ["Unknown"]

    def _generate_recommendations(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for preventing similar failures."""
        recommendations = []
        if isinstance(error, TimeoutError):
            recommendations.extend([
                "Increase step timeout values",
                "Check network connectivity",
                "Review LLM response times"
            ])
        elif isinstance(error, MemoryError):
            recommendations.extend([
                "Reduce batch sizes in data processing",
                "Implement data streaming for large datasets",
            ])
        elif isinstance(error, ConnectionError):
            recommendations.extend([
                "Implement retry logic with exponential backoff",
                "Check API rate limits",
            ])
        return recommendations

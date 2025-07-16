"""
TAgent Pipeline System

A sophisticated workflow orchestration feature that enables complex multi-step AI agent 
execution with memory persistence and inter-pipeline communication.
"""

from .models import PipelineStep, Pipeline, PipelineResult, ExecutionMode, DependencyType

__all__ = [
    "PipelineStep",
    "Pipeline", 
    "PipelineResult",
    "ExecutionMode",
    "DependencyType"
]
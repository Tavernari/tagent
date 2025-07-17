"""
Tests for the pipeline API.
"""

import pytest
from tagent.pipeline.api import (
    PipelineBuilder,
    PipelineOptimizer,
    PipelineSerializer,
    PipelineValidationError,
)
from tagent.pipeline.models import Pipeline, PipelineStep, ExecutionMode
from tagent.pipeline.conditions import ConditionDSL


class TestPipelineBuilder:
    """Tests for the PipelineBuilder."""

    def test_pipeline_builder_creates_pipeline(self):
        builder = PipelineBuilder("test_pipeline", "A test pipeline")
        builder.step("step1", "A test step")
        pipeline = builder.build()
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "test_pipeline"
        assert pipeline.description == "A test pipeline"

    def test_add_step(self):
        builder = PipelineBuilder("test_pipeline")
        builder.step("step1", "Goal for step 1")
        pipeline = builder.build()
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "step1"

    def test_parallel_steps(self):
        builder = PipelineBuilder("test_pipeline")
        builder.parallel_steps(
            {"name": "p_step1", "goal": "Parallel 1"},
            {"name": "p_step2", "goal": "Parallel 2"},
        )
        pipeline = builder.build()
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].execution_mode == ExecutionMode.CONCURRENT
        assert pipeline.steps[1].execution_mode == ExecutionMode.CONCURRENT

    def test_conditional_step(self):
        builder = PipelineBuilder("test_pipeline")
        condition = ConditionDSL.equals("a", "b")
        builder.conditional_step("cond_step", "A conditional step", condition)
        pipeline = builder.build()
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].condition == condition

    def test_validation_failure(self):
        builder = PipelineBuilder("test_pipeline")
        builder.step("step1", "Goal 1", depends_on=["non_existent"])
        with pytest.raises(PipelineValidationError):
            builder.build()

    def test_build_optimizes_pipeline(self):
        builder = PipelineBuilder("test_pipeline")
        builder.step("step2", "Goal 2", depends_on=["step1"])
        builder.step("step1", "Goal 1")
        pipeline = builder.build()
        # Optimizer should reorder steps
        assert pipeline.steps[0].name == "step1"
        assert pipeline.steps[1].name == "step2"

class TestPipelineOptimizer:
    """Tests for the PipelineOptimizer."""

    def test_optimize_execution_order(self):
        p = Pipeline("test")
        p.steps.append(PipelineStep(name="step3", goal="g", depends_on=["step2"]))
        p.steps.append(PipelineStep(name="step1", goal="g"))
        p.steps.append(PipelineStep(name="step2", goal="g", depends_on=["step1"]))
        
        optimizer = PipelineOptimizer(p)
        optimized_steps = optimizer.optimize_execution_order()
        
        optimized_names = [step.name for step in optimized_steps]
        assert optimized_names == ["step1", "step2", "step3"]

    def test_circular_dependency_detection(self):
        p = Pipeline("test")
        p.steps.append(PipelineStep(name="step1", goal="g", depends_on=["step2"]))
        p.steps.append(PipelineStep(name="step2", goal="g", depends_on=["step1"]))
        
        optimizer = PipelineOptimizer(p)
        with pytest.raises(PipelineValidationError):
            optimizer.optimize_execution_order()

    def test_identify_parallel_opportunities(self):
        p = Pipeline("test")
        p.steps.append(PipelineStep(name="step1", goal="g"))
        p.steps.append(PipelineStep(name="step2", goal="g"))
        p.steps.append(PipelineStep(name="step3", goal="g", depends_on=["step1", "step2"]))
        
        optimizer = PipelineOptimizer(p)
        opportunities = optimizer.identify_parallel_opportunities()
        
        assert len(opportunities) == 2
        assert sorted(opportunities[0]) == ["step1", "step2"]
        assert opportunities[1] == ["step3"]


class TestPipelineSerializer:
    """Tests for the PipelineSerializer."""

    @pytest.fixture
    def sample_pipeline(self):
        return (
            PipelineBuilder("serial_pipeline", "A test pipeline for serialization")
            .step("step1", "Goal 1")
            .step("step2", "Goal 2", depends_on=["step1"])
            .build()
        )

    def test_export_import_json(self, sample_pipeline):
        serializer = PipelineSerializer()
        json_data = serializer.export_pipeline(sample_pipeline, format='json')
        
        imported_pipeline = serializer.import_pipeline(json_data, format='json')
        
        assert imported_pipeline.name == sample_pipeline.name
        assert len(imported_pipeline.steps) == len(sample_pipeline.steps)
        assert imported_pipeline.steps[0].name == sample_pipeline.steps[0].name

    def test_export_import_yaml(self, sample_pipeline):
        serializer = PipelineSerializer()
        yaml_data = serializer.export_pipeline(sample_pipeline, format='yaml')
        
        imported_pipeline = serializer.import_pipeline(yaml_data, format='yaml')
        
        assert imported_pipeline.name == sample_pipeline.name
        assert len(imported_pipeline.steps) == len(sample_pipeline.steps)

    def test_export_import_pickle(self, sample_pipeline):
        serializer = PipelineSerializer()
        pickle_data = serializer.export_pipeline(sample_pipeline, format='pickle')
        
        imported_pipeline = serializer.import_pipeline(pickle_data, format='pickle')
        
        assert imported_pipeline.name == sample_pipeline.name
        assert len(imported_pipeline.steps) == len(sample_pipeline.steps)

    def test_unsupported_format(self, sample_pipeline):
        serializer = PipelineSerializer()
        with pytest.raises(ValueError):
            serializer.export_pipeline(sample_pipeline, format='xml')
        with pytest.raises(ValueError):
            serializer.import_pipeline("data", format='xml')


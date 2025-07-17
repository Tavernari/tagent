# 5. API Reference

This document provides a high-level overview of the most important classes you will interact with when building and running TAgent Pipelines.

## `PipelineBuilder`

The `PipelineBuilder` is the recommended way to construct a `Pipeline` object. It provides a fluent, chainable interface that makes the process of defining steps and their dependencies clear and readable.

**Location:** `tagent.pipeline.api.PipelineBuilder`

**Key Methods:**

- **`.step(name, goal, ...)`**: Adds a new step to the pipeline. This is the primary method you will use. It takes numerous optional arguments like `depends_on`, `execution_mode`, `timeout`, and `max_retries`.
- **`.build()`**: Finalizes the pipeline definition, runs validation and optimization, and returns a `Pipeline` object ready to be executed.

**Example Usage:**
```python
from tagent.pipeline.api import PipelineBuilder

builder = PipelineBuilder(name="MyPipeline")
pipeline = builder.step(...).step(...).build()
```

## `PipelineExecutor`

The `PipelineExecutor` is the engine that runs the pipeline. It takes a `Pipeline` object and a `TAgentConfig` object, which defines the default agent configuration for all steps.

**Location:** `tagent.pipeline.executor.PipelineExecutor`

**Key Methods:**

- **`__init__(self, pipeline, config)`**: The constructor requires the pipeline to run and the agent configuration.
- **`async execute()`**: The main method that starts the pipeline execution and returns a `PipelineResult` object once complete.

**Example Usage:**
```python
from tagent.pipeline.executor import PipelineExecutor
from tagent.config import TAgentConfig

# Assume 'pipeline' is a Pipeline object from the builder
agent_config = TAgentConfig(model="gpt-4")
executor = PipelineExecutor(pipeline, config=agent_config)
result = await executor.execute()
```

## `PipelineResult`

The `PipelineResult` is the object returned by the `PipelineExecutor` after the workflow finishes (either by succeeding or failing). It contains all the information about the execution.

**Location:** `tagent.pipeline.models.PipelineResult`

**Key Attributes:**

- **`.success` (bool)**: `True` if the entire pipeline completed successfully, `False` otherwise.
- **`.step_outputs` (dict)**: A dictionary where keys are the names of your steps and values are the results from each corresponding step. This is the most important attribute for retrieving the final data.
- **`.error_details` (dict)**: If the pipeline failed, this dictionary will contain information about which step failed and why.
- **`.execution_time` (float)**: The total time in seconds that the pipeline took to run.

**Example Usage:**
```python
# Assume 'result' is a PipelineResult object
if result.success:
    final_output = result.step_outputs.get("my_final_step_name")
    print(final_output.result)
```

---

This concludes the TAgent Pipelines documentation. You are now equipped with the knowledge to build, run, and debug complex AI workflows.

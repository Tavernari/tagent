# 5. API Reference

This document provides a high-level overview of the most important classes you will interact with when building and running TAgent Pipelines.

## `PipelineBuilder`

The `PipelineBuilder` is the recommended way to construct a `Pipeline` object. It provides a fluent, chainable interface that makes the process of defining steps and their dependencies clear and readable.

**Location:** `tagent.pipeline.api.PipelineBuilder`

**Key Methods:**

- **`.step(name, goal, ...)`**: Adds a new step to the pipeline. This is the primary method you will use. It takes numerous optional arguments like `depends_on`, `execution_mode`, `timeout`, `max_retries`, and `read_data`.
- **`.build()`**: Finalizes the pipeline definition, runs validation and optimization, and returns a `Pipeline` object ready to be executed.

**Example Usage:**
```python
from tagent.pipeline.api import PipelineBuilder

builder = PipelineBuilder(name="MyPipeline")
pipeline = builder.step(...).step(...).build()
```

### `.step()` Method Parameters

The `.step()` method accepts the following parameters:

- **`name`** (str): Unique identifier for the step
- **`goal`** (str): Description of what the step should accomplish
- **`depends_on`** (List[str], optional): List of step names this step depends on
- **`execution_mode`** (ExecutionMode, optional): `SERIAL` (default) or `CONCURRENT`
- **`timeout`** (int, optional): Maximum execution time in seconds
- **`max_retries`** (int, optional): Number of retry attempts on failure
- **`read_data`** (List[str], optional): List of data paths to read from previous steps
- **`tools`** (List[Callable], optional): Step-specific tools to use
- **`output_schema`** (BaseModel, optional): Pydantic model for structured output
- **`constraints`** (List[str], optional): Additional constraints for the step
- **`condition`** (Callable, optional): Condition function for conditional execution

### `read_data` Parameter

The `read_data` parameter enables sophisticated data flow control by allowing steps to read specific outputs from previous steps. This data is automatically injected into tool functions and added to the step's prompt context.

**Supported formats:**
- `"step_name"` - Read entire step output
- `"step_name.attribute"` - Read specific attribute from structured output
- `["step1.result", "step2.data"]` - Read multiple data paths

**Example:**
```python
from pydantic import BaseModel, Field

# Define the expected output structure
class AnalysisOutput(BaseModel):
    insights: str = Field(description="Key insights from the analysis")
    summary: str = Field(description="Summary of findings")
    metadata: dict = Field(description="Additional analysis metadata")

builder.step(
    name="analyze_data",
    goal="Analyze user feedback data.",
    output_schema=AnalysisOutput  # Define structured output with specific fields
).step(
    name="generate_report",
    goal="Generate a comprehensive report based on the analysis.",
    depends_on=["analyze_data"],
    read_data=["analyze_data.insights", "analyze_data.summary"],  # Reference defined fields
    tools=[report_generator]
)
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

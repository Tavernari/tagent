# 3. Core Pipeline Concepts

To build powerful and effective pipelines, it's important to understand the three fundamental concepts that govern how they work: Steps, Dependencies, and the flow of data (Context).

## 1. Pipeline Steps (`PipelineStep`)

A `PipelineStep` is the single most important building block of a pipeline. Each step is an independent TAgent task with its own unique configuration.

When you use the `PipelineBuilder`, each call to the `.step()` method creates a `PipelineStep` object.

```python
# This creates a PipelineStep object
builder.step(
    name="unique_step_name",  # A unique identifier for this step
    goal="The specific task for the agent in this step to accomplish.",
    # ... other configuration ...
)
```

The two most important attributes of a step are:
- **`name`**: A unique string that identifies the step within the pipeline. This is used to define dependencies and to access the step's output later.
- **`goal`**: The prompt or objective for the AI agent that will execute this step.

## 2. Dependencies (`depends_on`)

Dependencies define the order of execution. By making one step depend on another, you ensure that the parent step completes *before* the child step begins.

You define a dependency by passing a list of step names to the `depends_on` argument.

```python
builder.step(
    name="step_A",
    goal="This is the first step."
).step(
    name="step_B",
    goal="This step runs after step_A.",
    depends_on=["step_A"]  # step_B depends on step_A
).step(
    name="step_C",
    goal="This step runs after step_B.",
    depends_on=["step_B"]  # step_C depends on step_B
)
```

This creates the following execution graph:
**`step_A` → `step_B` → `step_C`**

The `PipelineExecutor` automatically analyzes this graph and runs the steps in the correct order.

## 3. Data Flow and Context

This is the most critical concept: **How does the output of one step become the input for another?**

The `PipelineExecutor` automatically manages this by injecting the results of all parent dependencies into the `goal` of the child step.

Let's look at our previous example: `step_A` → `step_B`.

1.  **`step_A` runs.** Let's say its goal was "Generate a random color" and its output (`result`) was `"blue"`.
2.  The `PipelineExecutor` stores this result in its memory: `{"step_A": "blue"}`.
3.  **`step_B` is ready to run.** Before executing it, the executor modifies its `goal` prompt.

    The original goal for `step_B` was:
    > "This step runs after step_A."

    The executor automatically appends the context from the completed dependencies. The new, modified goal becomes:
    > "This step runs after step_A.
    >
    > **Available data from dependencies:**
    > - **step_A**: blue"

The agent running in `step_B` now has the result from `step_A` directly in its prompt context, allowing it to use that information to accomplish its own goal.

This automatic context injection is the mechanism that allows you to chain tasks together to create sophisticated workflows.

---

With these concepts in mind, you are ready to explore more advanced features.

**➡️ [Next: Advanced Features](./04_advanced_features.md)**

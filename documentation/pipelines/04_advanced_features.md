# 4. Advanced Pipeline Features

Once you've mastered the core concepts, you can leverage TAgent's advanced features to build more efficient, robust, and dynamic pipelines.

## Step-Specific Tools

While you can provide a global list of tools in the `TAgentConfig`, you can also provide a specific list of tools for an individual step. This is useful for isolating functionality and improving security by ensuring a step only has access to the tools it needs.

Step-specific tools are **combined** with global tools. If a tool with the same name exists in both lists, the step-specific version will be used.

```python
# Define a global tool
def global_tool():
    """A tool available to all steps."""
    return "global_result", "from global"

# Define a step-specific tool
def specialized_tool():
    """A tool only for a specific step."""
    return "special_result", "from specialized"

# In the PipelineBuilder...
builder.step(
    name="special_step",
    goal="This step needs a special tool.",
    tools=[specialized_tool]  # Provide the tool directly to the step
)

# In the main run call...
agent_config = TAgentConfig(
    model="gpt-4o-mini",
    tools=[global_tool] # Global tools
)

executor = PipelineExecutor(pipeline, config=agent_config)
```
In this scenario, the `special_step` will have access to both `specialized_tool` and `global_tool`. Other steps in the pipeline will only have access to `global_tool`.

## Concurrent Execution

By default, steps run in `SERIAL` mode. However, you can run independent steps in `CONCURRENT` mode to save time.

In this example, we'll research a topic and then *simultaneously* write a blog post and a tweet about it.

```python
# In the PipelineBuilder...
builder.step(
    name="research",
    goal="Research the benefits of a 4-day work week."
).step(
    name="write_blog_post",
    goal="Write a 300-word blog post based on the research.",
    depends_on=["research"],
    execution_mode=ExecutionMode.CONCURRENT # This can run in parallel
).step(
    name="write_tweet",
    goal="Write a catchy tweet based on the research.",
    depends_on=["research"],
    execution_mode=ExecutionMode.CONCURRENT # This can also run in parallel
).step(
    name="final_summary",
    goal="Combine the blog post and the tweet into a final summary document.",
    # This step depends on BOTH parallel steps, creating a "fan-in"
    depends_on=["write_blog_post", "write_tweet"]
)
```

The execution graph looks like this:

```
           /--> write_blog_post --\
research --<                        >--> final_summary
           \--> write_tweet   ----/
```

The `PipelineExecutor` will automatically run `write_blog_post` and `write_tweet` at the same time after `research` is complete, potentially cutting down the total execution time.

*Note: You must import `ExecutionMode` from `tagent.pipeline.models`.*

## Timeouts and Retries

You can make your pipeline more robust by setting timeouts and retry policies for steps that might be slow or fail intermittently.

```python
# In the .step() definition...
builder.step(
    name="fetch_external_api",
    goal="Fetch data from a potentially unreliable external API.",
    timeout=120,       # Abort this step if it takes longer than 120 seconds
    max_retries=3      # If the step fails, retry it up to 3 times
)
```

The `PipelineExecutor` will automatically manage the retry logic. If the step fails more than `max_retries`, the entire pipeline will be marked as failed.

## Conditional Execution (Coming Soon)

The ability to execute or skip steps based on the output of previous steps is a planned feature. The internal architecture includes placeholders for a `condition` attribute on a `PipelineStep`.

The envisioned API would look something like this:

```python
# --- THIS IS A FUTURE EXAMPLE, NOT YET IMPLEMENTED ---
from tagent.pipeline.conditions import IsGreaterThan

builder.step(
    name="check_sentiment",
    goal="Analyze the sentiment of a user review and return a score from 0 to 10."
).step(
    name="send_thank_you_email",
    goal="Draft a thank you email for the positive review.",
    depends_on=["check_sentiment"],
    # This step would only run if the output of 'check_sentiment' is greater than 7
    condition=IsGreaterThan(step="check_sentiment", value=7)
).step(
    name="escalate_to_support",
    goal="Draft a support ticket to address the negative review.",
    depends_on=["check_sentiment"],
    # This step would only run if the output is NOT greater than 7
    condition=Not(IsGreaterThan(step="check_sentiment", value=7))
)
```
This would allow for dynamic, branching logic within your workflows.

---

Finally, let's look at the key classes that make all of this possible.

➡️ [Next: API Reference](./05_api_reference.md)

```
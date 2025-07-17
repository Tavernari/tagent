# 2. Getting Started: Your First Pipeline

This tutorial will guide you through creating and running a simple, two-step pipeline. Our goal is to build a workflow that first generates a creative topic and then writes a tagline for it.

## Prerequisites

Ensure you have TAgent installed with the pipeline extra:
```bash
pip install "tagent[pipeline]"
```

## The Complete Script

Create a Python file named `run_first_pipeline.py` and paste the following code into it. Each part of the script is explained below.

```python
import asyncio
from tagent.pipeline.api import PipelineBuilder
from tagent.pipeline.executor import PipelineExecutor
from tagent.config import TAgentConfig

async def main():
    """
    This function defines, configures, and runs our pipeline.
    """
    print("🚀 Starting our first TAgent Pipeline...")

    # 1. Define the Pipeline using the PipelineBuilder
    # The builder provides a "fluent" interface to chain steps together.
    builder = PipelineBuilder(
        name="Creative Content Pipeline",
        description="A simple pipeline to generate a topic and a tagline."
    )

    pipeline = builder.step(
        name="topic_generation",
        goal="Generate a single, interesting and unexpected topic for a tech blog post. For example: 'The Philosophy of Underwater Basket Weaving in the Digital Age'."
    ).step(
        name="tagline_creation",
        goal="Create a catchy, one-sentence tagline for the blog post topic provided by the previous step.",
        depends_on=["topic_generation"]  # This is the key: it creates the dependency!
    ).build()

    # 2. Configure the Agent
    # This TAgentConfig will be used by the agent running in each step.
    agent_config = TAgentConfig(
        model="gpt-4o-mini", # Feel free to use another model
        verbose=False      # We set this to False to keep the pipeline output clean
    )

    # 3. Create and Run the Executor
    # The executor takes the pipeline definition and the agent config.
    executor = PipelineExecutor(pipeline, config=agent_config)
    result = await executor.execute()

    # 4. Inspect the Results
    if result.success:
        print("\n✅ Pipeline completed successfully!")

        # The results of each step are stored in `result.step_outputs`
        # We access the output by the step's name.
        topic_output = result.step_outputs.get("topic_generation")
        tagline_output = result.step_outputs.get("tagline_creation")

        if topic_output:
            # The actual content is in the 'result' attribute of the default output model
            print(f"\n🧠 Generated Topic: {topic_output.result}")
        if tagline_output:
            print(f"✨ Generated Tagline: {tagline_output.result}")

    else:
        print("\n❌ Pipeline failed.")
        print(f"Error Details: {result.error_details}")

if __name__ == "__main__":
    # Run the asynchronous main function
    asyncio.run(main())

```

## How to Run It

Save the code above and run it from your terminal:

```bash
python run_first_pipeline.py
```

## Expected Output

You will see output from the `PipelineExecutor` as it runs each step. The final output should look something like this:

```
🚀 Starting our first TAgent Pipeline...
... (Executor logs) ...

✅ Pipeline completed successfully!

🧠 Generated Topic: The Ethics of AI-Generated Art: Can a Machine Be a Muse?
✨ Generated Tagline: Where code creates, and consciousness questions.
```

Congratulations! You've successfully built and executed a multi-step AI workflow.

---

Now that you've seen a pipeline in action, let's dive deeper into the fundamental concepts.

**➡️ [Next: Core Concepts](./03_core_concepts.md)**

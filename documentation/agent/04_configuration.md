# 4. Configuration

TAgent provides a flexible configuration system that allows you to manage API keys, select models, and control agent behavior.

## Managing API Keys

The recommended way to manage API keys is through a `.env` file in the root of your project. TAgent automatically loads environment variables from this file.

Create a file named `.env`:
```
# For OpenAI
OPENAI_API_KEY="sk-..."

# For Google
GEMINI_API_KEY="..."

# For Anthropic
ANTHROPIC_API_KEY="..."

# For any other provider supported by LiteLLM, use their environment variable
# For example, for OpenRouter:
# OPENROUTER_API_KEY="..."
```

You can also set environment variables directly in your script before running the agent, but this is less secure.
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"
```

## Using the `TAgentConfig` Object

For more advanced control, you can pass a `TAgentConfig` object to the `run_agent` function.

```python
import asyncio
from tagent import run_agent
from tagent.config import TAgentConfig

async def main():
    # Create a configuration object
    config = TAgentConfig(
        model="gpt-4o-mini",
        max_iterations=5,  # Stop the agent after 5 planning/execution cycles
        verbose=True,
        # You can also pass the API key directly here, though .env is preferred
        # api_key="your-key-here"
    )

    await run_agent(
        goal="A simple goal.",
        config=config
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuring Different Models for Different Tasks

You can specify different models for the different phases of the agent's state machine. This can be useful for saving costs, for example, by using a smaller, faster model for execution and a larger, more powerful model for final evaluation.

```python
from tagent.model_config import AgentModelConfig

# Define a multi-model configuration
model_config = AgentModelConfig(
    # The main model, used as a default if others aren't set
    tagent_model="gpt-4o-mini",

    # Model for the PLANNING phase
    planner_model="gpt-4-turbo",

    # Model for the EXECUTING phase (can be smaller/faster)
    executor_model="gpt-3.5-turbo",

    # Model for the final EVALUATING phase (should be powerful)
    evaluator_model="gpt-4-turbo",
)

# Pass this object to the `model` parameter
await run_agent(
    goal="A complex goal requiring careful planning and evaluation.",
    model=model_config
)
```

This concludes the documentation for the core TAgent. You now have the knowledge to run agents, provide them with tools, and configure them to your needs.

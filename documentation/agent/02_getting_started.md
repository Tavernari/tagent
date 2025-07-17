# 2. Getting Started with TAgent

This tutorial will walk you through the simplest way to use TAgent: running a single agent to achieve a specific goal without any custom tools.

## Prerequisites

Ensure you have TAgent installed.
```bash
pip install tagent
```
You will also need an API key for an LLM provider (like OpenAI, Google, Anthropic, etc.). TAgent uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, so it supports hundreds of models.

## Your First Script

Create a Python file named `run_basic_agent.py`.

```python
import os
from tagent import run_agent

# It's good practice to manage your API key via environment variables.
# TAgent automatically loads them from a .env file in your project root.
# For this example, you can also set it directly.
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

async def main():
    """
    This function defines the goal and runs the agent.
    """
    print("🚀 Running our first TAgent...")

    # Define the goal for the agent
    goal = "Translate the phrase 'Hello, world!' into Japanese and also provide the romanized (romaji) version of the translation."

    # Run the agent
    # - `goal`: The objective for the agent.
    # - `model`: The LLM to use. Make sure the model name matches what your provider expects.
    # - `verbose`: Set to True to see the agent's thought process (planning, execution steps, etc.).
    result = await run_agent(
        goal=goal,
        model="gpt-4o-mini", # Or "gemini/gemini-1.5-pro", "claude-3-haiku-20240307", etc.
        verbose=True
    )

    # Inspect the final result
    print("\n" + "="*30)
    if result.goal_achieved:
        print("✅ Goal Achieved!")
        # The final, structured output is in the `output` attribute.
        # For the default agent, the main answer is in `output.result`.
        print(f"Final Answer: {result.output.result}")
    else:
        print("❌ Goal Not Achieved.")
        print(f"Failure Reason: {result.failure_reason}")
    print("="*30)


if __name__ == "__main__":
    import asyncio
    # TAgent's `run_agent` is an async function, so we run it in an event loop.
    asyncio.run(main())
```

## How to Run It

1.  **Set your API key.** You can either create a `.env` file in the same directory with the line `OPENAI_API_KEY=sk-...` (or the equivalent for your provider) or uncomment the `os.environ` line in the script and paste your key there.
2.  **Run the script from your terminal:**
    ```bash
    python run_basic_agent.py
    ```

## Expected Output

Because `verbose=True`, you will see the agent's internal state as it plans and executes its tasks. The final output at the end should look like this:

```
==============================
✅ Goal Achieved!
Final Answer: The Japanese translation of 'Hello, world!' is 'こんにちは、世界！' (Konnichiwa, Sekai!).
==============================
```

You have successfully run your first AI agent!

---

Next, let's see how to give your agent new capabilities by providing it with custom tools.

**➡️ [Next: Using Tools](./03_using_tools.md)**

```
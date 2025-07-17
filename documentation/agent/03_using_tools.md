# 3. Using Tools

The real power of TAgent comes from its ability to use custom tools. A "tool" is simply a Python function that you provide to the agent, giving it new capabilities beyond the knowledge of the LLM.

## How TAgent Handles Tools

TAgent's `ToolExecutor` is designed to be flexible. Instead of forcing you to write functions with a rigid signature like `def my_tool(state, args)`, it intelligently inspects your function's signature and provides only the arguments it needs.

This allows you to write natural, Pythonic code.

## Example: A Simple Web Search Tool

Let's create a tool that simulates a web search.

```python
import asyncio
from tagent import run_agent
from pydantic import BaseModel, Field

# 1. Define a Pydantic model for structured arguments (optional but recommended)
# This gives the LLM a clear schema of the arguments your tool expects.
class SearchArgs(BaseModel):
    query: str = Field(description="The specific query for the web search.")

# 2. Write the tool function
# This function takes the Pydantic model as an argument. TAgent will automatically
# populate it from the LLM's structured output.
def web_search(args: SearchArgs):
    """
    A tool that simulates a web search and returns a string of results.
    The docstring is very important, as it's what the LLM sees to understand what the tool does.
    """
    print(f"--- TOOL: Performing web search for '{args.query}' ---")
    # In a real scenario, you would use a library like `requests` or `beautifulsoup` here.
    return "search_results", f"Simulated search results for the query: '{args.query}'"

async def main():
    # 3. Provide the tool to the agent
    # The `tools` argument is a list of tool functions.
    # TAgent will use the function's __name__ as the tool name.
    tools = [web_search]

    goal = "Find out what the TAgent framework is and what it's used for."

    result = await run_agent(
        goal=goal,
        tools=tools,
        model="gpt-4o-mini",
        verbose=True
    )

    print("\n" + "="*30)
    if result.goal_achieved:
        print("✅ Goal Achieved!")
        print(f"Final Answer: {result.output.result}")
    else:
        print("❌ Goal Not Achieved.")
    print("="*30)

if __name__ == "__main__":
    asyncio.run(main())
```

When you run this script, you will see in the verbose logs that the agent:
1.  **Plans** to use the `web_search` tool.
2.  **Executes** the tool with a relevant query (e.g., `{"query": "TAgent framework"}`).
3.  Prints our custom message "--- TOOL: Performing web search... ---".
4.  Receives the tool's output and uses it to formulate the final answer.

## Flexible Signatures

TAgent supports other signatures too:

- **Simple Keyword Arguments**:
  ```python
  def get_weather(city: str, unit: str = "celsius"):
      # ...
  ```
- **Accessing Agent State**: If your tool needs to read or modify the agent's central state, add a `state` parameter.
  ```python
  from typing import Dict, Any

  def my_stateful_tool(state: Dict[str, Any], args: MyArgs):
      # Access the database connection stored in the state
      db_conn = state.get("db_connection")
      # ...
  ```

---

➡️ [Next: Configuration](./04_configuration.md)

```
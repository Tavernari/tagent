# TAgent: Build Powerful AI Agents, Not Boilerplate

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.6.2-green.svg)](https://github.com/yourusername/tagent2)

> **A flexible, task-based framework for creating powerful AI agents with minimal code. No course required.**

![TAgent in Action](examples/tab_news_analyzer/tabnews_code_example.gif)

Tired of frameworks that require more time learning their abstractions than building your solution? TAgent is designed for developers who want to build powerful, task-oriented AI agents without the unnecessary complexity. Write simple Python functions, and let our intelligent `ToolExecutor` handle the rest.

## Why TAgent?

TAgent follows a simple philosophy: **task-based execution with intelligent fallbacks**. Instead of complex function calling or massive dependency trees, you get:

- **🎯 Task-Based Architecture**: Clear phase-based execution (INIT → PLAN → EXECUTE → EVALUATE → FINALIZE)
- **🔄 Retry Logic**: Automatic retry for failed tasks with intelligent fallbacks
- **🧠 LLM Fallbacks**: When tools aren't available, uses LLM knowledge directly
- **📚 Enhanced Memory**: Simple context management for better decision making
- **📊 Structured Outputs**: Works with any LLM via JSON, not function calling
- **⚡ Zero Boilerplate**: Get started with 3 lines of code

## Quick Start

```bash
pip install -e .
```

```python
from tagent import run_agent

# Simple usage - task-based approach
result = run_agent(
    goal="Translate 'Hello world' to Chinese",
    model="gpt-4",
    verbose=True
)

print(f"Goal achieved: {result.goal_achieved}")
print(f"Final output: {result.output.result}")
# The type of 'result.output' is automatically inferred by static analysis tools!
# Output: Goal achieved: True
# Final output: 你好世界
```

### With Custom Tools

```python
from tagent import run_agent
from pydantic import BaseModel, Field

# Define a custom tool
def search_web(state, args):
    query = args.get("query", "")
    # Simulate web search
    return ("search_results", f"Results for: {query}")

# Define output format
class SearchReport(BaseModel):
    query: str = Field(description="The search query")
    results: str = Field(description="Search results summary")

# Run agent with tools
result = run_agent(
    goal="Search for information about Python async programming",
    tools={"search_web": search_web},
    output_format=SearchReport,
    model="gpt-4",
    verbose=True
)

print(f"Tasks completed: {result.completed_tasks}")
print(f"Final output: {result.output}")
```

## 🏗️ Task-Based Architecture

TAgent v0.6.2 introduces a revolutionary **task-based approach** that makes AI agent behavior predictable and reliable:

### The Flow

```
INIT → PLANNING → EXECUTING → EVALUATING → FINALIZE
  ↓        ↓         ↓           ↓           ↓
Setup   Create    Execute     Check if    Format
        Tasks     Tasks       Goal Met    Output
                   ↓
              (Retry failed tasks)
                   ↓
              (Return to PLANNING if needed)
```

### Key Features

#### 1. **Context Engineering with RAG**
TAgent leverages a powerful **Retrieval-Augmented Generation (RAG)** system to provide the LLM with relevant, just-in-time context. This system dynamically pulls information from three key sources:
- **Instructions**: High-level guidance on how the agent should behave, plan, and execute tasks.
- **Tools**: The full documentation of all available tools, including their descriptions, parameters, and examples.
- **Memories**: Learnings from past actions, including successes, failures, and key data points.

This ensures the agent always has the information it needs to make optimal decisions without overwhelming the context window.

#### 2. **Intelligent Planning**
- Creates specific, actionable tasks to achieve the goal
- Considers available tools and previous failures
- Uses RAG context for better decision making

#### 3. **Robust Execution**
- Executes tasks one by one with retry logic (3 retries by default)
- **LLM Fallback**: When tools aren't available, uses LLM knowledge directly
- Intelligent failure recovery and re-planning

#### 4. **Smart Evaluation**
- Assesses goal achievement after task completion
- Provides detailed feedback on what's missing
- Decides whether to retry or proceed to finalization

#### 5. **Comprehensive Finalization**
- Creates structured final output based on specified format
- Includes execution statistics and context
- Provides clear results for the user

## 🧠 Enhanced Memory System (Simple Context Management)

TAgent includes a **simple context management** system that helps the agent make better decisions:

### How Context Management Works

1. **Memory Storage**: Automatically stores important facts, execution results, and learned patterns in memory
2. **Context Retrieval**: Provides relevant context for each phase using keyword-based search
3. **Decision Enhancement**: Uses stored context to improve planning and execution decisions

### Memory Types

```python
# Automatically stored memories include:
- Execution results (success/failure)
- Key facts learned during execution
- Tool usage patterns
- Error patterns and solutions
- Goal-specific insights
```

### Context Usage

```python
# The system automatically uses context for:
- Better task planning based on previous experiences
- Smarter tool selection and parameter choices
- Improved error handling and recovery
- Enhanced goal evaluation
```

### Note on Implementation

The current implementation uses simple keyword-based search in memory. For production use cases requiring semantic search, you can extend the system with:
- Vector embeddings (OpenAI, Sentence Transformers)
- Vector databases (Chroma, Pinecone, Weaviate)
- Semantic similarity search

## 🛠️ Intelligent Tool System

TAgent features a smart `ToolExecutor` that adapts to your tools, not the other way around. This eliminates boilerplate and allows you to write natural, Pythonic functions.

### Flexible Tool Signatures

Instead of forcing a rigid `(state, args)` signature, you can define tools with parameters that make sense for the task. The `ToolExecutor` automatically inspects the function signature and provides the correct arguments.

**Supported Parameter Types:**

1.  **`state: Dict[str, Any]`**: If your tool needs access to the agent's central state, simply add a `state` parameter.
2.  **Pydantic Models**: For structured, validated inputs, define a Pydantic model and use it as a type hint. The executor will automatically instantiate it from the LLM's arguments.
3.  **Simple Keyword Arguments**: Any other keyword arguments (e.g., `query: str`, `user_id: int`) will be automatically populated from the LLM's output.

**Example:**

```python
from pydantic import BaseModel, Field
from typing import Dict, Any

# A Pydantic model for structured arguments
class UserProfile(BaseModel):
    user_id: int = Field(description="The user's unique identifier")
    new_email: str = Field(description="The new email address to set")

# The tool function with a flexible signature
def update_user_email(state: Dict[str, Any], profile: UserProfile):
    """Updates a user's email address in the system."""
    
    # Access agent state
    db_connection = state.get("db_connection")
    
    # Use validated Pydantic model
    print(f"Updating user {profile.user_id} with new email {profile.new_email}")
    
    # ... tool logic ...
    
    return "update_status", {"success": True, "user_id": profile.user_id}
```

### Smart Return Values

The `ToolExecutor` also handles return values intelligently:

1.  **`(key, value)` Tuple**: Return a tuple to explicitly define the key for the state update.
2.  **Pydantic Model**: Return a Pydantic model, and the executor will use its class name as the key (e.g., `UserProfile()` becomes `("userprofile", ...)`.
3.  **Other Types (`str`, `dict`, `list`)**: For any other return type, the key will be inferred from the tool's name (e.g., a tool named `search_web` returns `("search_web_output", ...)`).
4.  **`None`**: If a tool returns `None`, it's treated as a "fire-and-forget" operation, and no state update occurs.

This system makes defining and using tools incredibly intuitive and flexible.

## 📊 Result Structure

TAgent returns a comprehensive `TaskBasedAgentResult` object:

```python
class TaskBasedAgentResult(BaseModel):
    output: Any                      # Your structured result, with the correct type hint
    goal_achieved: bool             # Success indicator
    iterations_used: int            # Execution steps taken
    planning_cycles: int            # Planning iterations
    total_tasks: int                # Total tasks created
    completed_tasks: int            # Successfully completed
    failed_tasks: int               # Failed tasks
    state_summary: Dict[str, Any]   # Execution state
    memory_summary: Dict[str, Any]  # Context system summary
    failure_reason: Optional[str]   # Failure details if any
```

### Default Output Format

When no `output_format` is specified, TAgent returns:

```python
class DefaultFinalOutput(BaseModel):
    result: str                     # Main answer for the user
    summary: str                    # Execution summary
    achievements: List[str]         # What was accomplished
    challenges: List[str]           # Issues encountered
    data_collected: Dict[str, Any]  # All collected data
```

## 🔧 Configuration System

### Basic Configuration

```python
from tagent import run_agent
from tagent.config import TAgentConfig
from tagent.ui.factory import UIStyle

# Create configuration
config = TAgentConfig(
    model="gpt-4o-mini",
    max_iterations=10,
    verbose=True,
    ui_style=UIStyle.INSTITUTIONAL,
    api_key="your-api-key"
)

# Use configuration
result = run_agent("Your goal here", config=config)
```

### Environment Variables

TAgent automatically loads environment variables from `.env` files:

```bash
# .env file
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key
OPENROUTER_API_KEY=your-openrouter-key
```

### Model Configuration

```python
# Simple model selection
result = run_agent(
    goal="Your goal",
    model="gpt-4",  # or "claude-3-sonnet", "gemini-pro", etc.
)

# Advanced model configuration
from tagent.model_config import AgentModelConfig

config = AgentModelConfig(
    tagent_model="gpt-4",
    planner_model="gpt-4",
    executor_model="gpt-3.5-turbo",
    evaluator_model="gpt-4",
    api_key="your-key"
)

result = run_agent(goal="Your goal", model=config)
```

## 🎨 UI Styles

TAgent includes a modern, professional UI style inspired by VSCode:

```python
from tagent.ui.factory import UIStyle

# Choose your style
result = run_agent(
    goal="Your goal",
    ui_style=UIStyle.MODERN    # The default, modern, professional look
    # or UIStyle.INSTITUTIONAL   # Clean, logging-focused output
)
```

## 📚 Example

### Simple Translation

```python
from tagent import run_agent

result = run_agent(
    goal="Translate 'Hello world' to Chinese",
    model="gpt-4"
)

print(f"🎯 RESULT: {result.output.result}")
print(f"📝 SUMMARY: {result.output.summary}")
```

## 🚀 Advanced Features

### Retry Logic

```python
# Built-in retry for failed tasks
result = run_agent(
    goal="Fetch data from unreliable API",
    tools={"fetch_api": unreliable_api_tool},
    model="gpt-4"
    # Automatically retries failed tasks up to 3 times
)
```

### Planning Cycles

```python
# Intelligent re-planning when tasks fail
result = run_agent(
    goal="Complex multi-step task",
    tools=tools,
    model="gpt-4",
    max_iterations=20,
    # Will re-plan up to 5 times if needed
)
```

### Memory Management

```python
# Simple context system automatically manages memory
result = run_agent(
    goal="Learn from previous iterations",
    tools=tools,
    model="gpt-4"
    # Memories are automatically stored and retrieved using keyword search
)

# Access memory summary
print(f"Memories stored: {result.memory_summary}")
```

## 🔍 Debugging and Monitoring

### Verbose Mode

```python
result = run_agent(
    goal="Your goal",
    model="gpt-4",
    verbose=True  # Shows detailed execution logs
)
```

### Execution Statistics

```python
# Get detailed execution info
print(f"Planning cycles: {result.planning_cycles}")
print(f"Total tasks: {result.total_tasks}")
print(f"Completed: {result.completed_tasks}")
print(f"Failed: {result.failed_tasks}")
print(f"Iterations: {result.iterations_used}")
```

### State Inspection

```python
# Access full execution state
print(f"State summary: {result.state_summary}")
print(f"Memory summary: {result.memory_summary}")
```

## 🛡️ Error Handling

TAgent includes robust error handling:

- **Tool Failures**: Automatic retry with exponential backoff
- **LLM Failures**: Fallback strategies and graceful degradation
- **Network Issues**: Timeout handling and retry logic
- **Planning Failures**: Re-planning with failure context
- **Memory Issues**: Automatic memory cleanup and optimization

## 🎯 Best Practices

### 1. Tool Design

```python
# ✅ Good tool design
def good_tool(state, args):
    """Clear description of what the tool does."""
    try:
        # Validate inputs
        required_param = args.get("required_param")
        if not required_param:
            return ("error", {"message": "required_param is missing"})
        
        # Do work
        result = do_work(required_param)
        
        # Return tuple
        return ("result_key", result)
    except Exception as e:
        return ("error", {"message": str(e)})

# ❌ Avoid this
def bad_tool(state, args):
    # No documentation, no error handling
    return process_data(args["data"])
```

### 2. Goal Definition

```python
# ✅ Clear, specific goals
goal = "Extract the latest 5 articles from TechCrunch, summarize each, and translate summaries to Spanish"

# ❌ Vague goals
goal = "Do something with articles"
```

### 3. Output Formats

```python
# ✅ Well-defined output structure
class ArticleReport(BaseModel):
    title: str = Field(description="Article title")
    summary: str = Field(description="Brief summary")
    url: str = Field(description="Article URL")
    published_date: str = Field(description="Publication date")

# ❌ Generic output
class GenericOutput(BaseModel):
    data: Any
```

## 🤝 Contributing

We welcome contributions! See our [contributing guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with ❤️ using:
- [LiteLLM](https://github.com/BerriAI/litellm) for LLM integration
- [Pydantic](https://pydantic.dev/) for data validation

---

**TAgent v0.6.2** - Simplifying AI for your daily life 🚀
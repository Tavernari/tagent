# TAgent - When You're Tired of Unnecessarily Complex Agent Frameworks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.5.6-green.svg)](https://github.com/yourusername/tagent2)

> **A task-based, minimalist framework for AI agents that actually makes sense**

Fed up with bloated frameworks that need 50 dependencies and 200 lines of boilerplate just to make a simple automation? TAgent is a straightforward, task-based approach to building AI agents that solve specific problems without the unnecessary complexity.

![gif](https://vhs.charm.sh/vhs-dujKmiVTP09yg9gOXAbs5.gif)

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
print(f"Final output: {result.final_output.result}")
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
print(f"Final output: {result.final_output}")
```

## 🏗️ Task-Based Architecture

TAgent v0.5.6 introduces a revolutionary **task-based approach** that makes AI agent behavior predictable and reliable:

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

#### 1. **Intelligent Planning**
- Creates specific, actionable tasks to achieve the goal
- Considers available tools and previous failures
- Uses RAG context for better decision making

#### 2. **Robust Execution**
- Executes tasks one by one with retry logic (3 retries by default)
- **LLM Fallback**: When tools aren't available, uses LLM knowledge directly
- Intelligent failure recovery and re-planning

#### 3. **Smart Evaluation**
- Assesses goal achievement after task completion
- Provides detailed feedback on what's missing
- Decides whether to retry or proceed to finalization

#### 4. **Comprehensive Finalization**
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

## 🛠️ Tool System

### Tool Interface

All tools follow a simple, consistent interface:

```python
def my_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Tool description.
    
    Args:
        state: Current agent state for context
        args: Tool-specific arguments
        
    Returns:
        Tuple of (key, value) for state update, or None if failed
    """
    # Tool logic here
    return ("result_key", result_value)
```

### LLM Fallback System

When tools aren't available, TAgent automatically uses the LLM as a fallback:

```python
# No tools? No problem!
result = run_agent(
    goal="Translate 'Hello world' to Chinese",
    model="gpt-4"
    # No tools defined - will use LLM fallback
)
# Still works perfectly! 🎉
```

### Tool Registration

```python
# Register multiple tools
tools = {
    "web_search": search_web_tool,
    "file_read": read_file_tool,
    "calculate": math_tool,
    "translate": translate_tool
}

result = run_agent(
    goal="Research AI trends and create a report",
    tools=tools,
    model="gpt-4"
)
```

## 📊 Result Structure

TAgent returns a comprehensive `TaskBasedAgentResult` object:

```python
class TaskBasedAgentResult(BaseModel):
    final_output: Any               # Your structured result
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

TAgent includes beautiful, retro-inspired UI styles:

```python
from tagent.ui.factory import UIStyle

# Choose your style
result = run_agent(
    goal="Your goal",
    ui_style=UIStyle.ANIMATED    # Animated progress bars
    # or UIStyle.INSTITUTIONAL   # Clean, professional look
)
```

## 📚 Examples

### Simple Translation

```python
from tagent import run_agent

result = run_agent(
    goal="Translate 'Hello world' to Chinese",
    model="gpt-4"
)

print(f"🎯 RESULT: {result.final_output.result}")
print(f"📝 SUMMARY: {result.final_output.summary}")
```

### Web Scraping and Analysis

```python
from tagent import run_agent
from pydantic import BaseModel, Field

# Define tools
def extract_articles(state, args):
    # Extract articles from RSS
    return ("articles", articles_list)

def analyze_content(state, args):
    # Analyze article content
    return ("analysis", analysis_result)

# Define output format
class ArticleAnalysis(BaseModel):
    title: str = Field(description="Article title")
    summary: str = Field(description="Article summary")
    sentiment: str = Field(description="Sentiment analysis")

# Run agent
result = run_agent(
    goal="Extract latest tech articles and analyze sentiment",
    tools={
        "extract_articles": extract_articles,
        "analyze_content": analyze_content
    },
    output_format=ArticleAnalysis,
    model="gpt-4"
)
```

### Complex Research Task

```python
from tagent import run_agent

# Define research tools
tools = {
    "search_web": web_search_tool,
    "read_pdf": pdf_reader_tool,
    "analyze_data": data_analysis_tool,
    "create_chart": visualization_tool
}

result = run_agent(
    goal="Research AI market trends for 2024 and create a comprehensive report",
    tools=tools,
    model="gpt-4",
    max_iterations=50,
    verbose=True
)

print(f"Research completed: {result.goal_achieved}")
print(f"Tasks executed: {result.completed_tasks}")
print(f"Final report: {result.final_output}")
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
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output

---

**TAgent v0.5.6** - Making AI agents simple again! 🚀
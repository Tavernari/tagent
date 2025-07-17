# TAgent: Build Powerful AI Agents, Not Boilerplate

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.7.1-green.svg)](https://github.com/yourusername/tagent2)

**A developer-first framework for crafting everything from simple AI assistants to complex, multi-agent workflows with elegance and ease.**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TAgent Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Simple Agent (run_agent)           Multi-Step Pipeline System                  │
│  ┌─────────────────────┐            ┌─────────────────────────────────────────┐ │
│  │                     │            │                                         │ │
│  │   Goal → Agent      │            │    Step 1 → Step 2 → Step 3             │ │
│  │      ↓              │            │      ↓       ↓       ↓                  │ │
│  │   [Tools]           │     →      │   [Tools]  [Tools]  [Tools]             │ │
│  │      ↓              │            │      ↓       ↓       ↓                  │ │
│  │    Result           │            │   Conditional Execution & Dependencies  │ │
│  │                     │            │      ↓       ↓       ↓                  │ │
│  └─────────────────────┘            │   Structured Outputs & State Flow       │ │
│                                     │                                         │ │
│  Perfect for:                       │  Perfect for:                           │ │
│  • Quick tasks                      │  • Complex workflows                    │ │
│  • Single-step operations           │  • Multi-step processes                 │ │
│  • Simple automations               │  • Conditional logic                    │ │
│                                     │  • Parallel execution                   │ │
│                                     │                                         │ │
│                                     └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Why TAgent?

-   **Focus on Your Logic**: Stop wrestling with complex frameworks. Write standard Python functions, and TAgent's `ToolExecutor` intelligently adapts to them.
-   **Build, Don't Just Prompt**: Move beyond simple prompting. Create robust, stateful agents that can plan, execute, and learn from their actions.
-   **Scale with Confidence**: Start with a single agent for a simple task. As your needs grow, scale up to a multi-step, parallelized workflow with the powerful **Pipelines** engine.
-   **Universal Compatibility**: TAgent is model-agnostic. By using structured JSON outputs instead of proprietary function-calling, it works with hundreds of LLMs out-of-the-box.

## Features at a Glance

-   **🧠 Task-Based Agents**: Predictable, state-driven agents that plan and execute to achieve goals.
-   **🛠️ Developer-First Tools**: Your Python functions are first-class citizens. No boilerplate required.
-   **🚀 Powerful Pipeline Engine**: Orchestrate complex, multi-step workflows with dependencies, parallelism, and advanced data flow control.
-   **🔗 Smart Data Injection**: Use `read_data` to automatically inject outputs from previous steps as tool parameters, enabling sophisticated prompt enhancement.
-   **🌐 Model Agnostic**: Compatible with any LLM provider, including OpenAI, Google, Anthropic, and more via LiteLLM.
-   **🔒 Structured & Reliable**: Enforces structured outputs for predictable, type-safe results using Pydantic.
-   **🤖 Simple & Scalable**: Start with a few lines of code and grow to production-grade automations.

---

## Quick Look

### Simple Agent Tasks

See how easy it is to create an agent with custom tools for simple tasks.

```python
from tagent import run_agent

# 1. Give your agent a goal
goal = "What is the current stock price for NVDA and should I buy it?"

# 2. Give it a tool (a simple Python function)
def get_stock_price(symbol: str):
    """A tool to get the latest stock price for a stock symbol."""
    print(f"--- Getting price for {symbol} ---")
    # (Your logic to call a real stock API would go here)
    if symbol == "NVDA":
        return "stock_price", {"symbol": "NVDA", "price": 950.00}
    return "stock_price", {"symbol": symbol, "price": "unknown"}

# 3. Run the agent
result = run_agent(
    goal=goal,
    tools=[get_stock_price],
    model="gpt-4o-mini"
)

print(result.final_output)
```

### Multi-Step Workflows with Pipelines

For more complex scenarios, use TAgent's Pipeline system to build sophisticated workflows:

```python
from tagent.pipeline import PipelineBuilder
from tagent.pipeline.conditions import IsGreaterThan, IsLessThan

from tagent.pipeline.executor import PipelineExecutor, PipelineExecutorConfig
from tagent.config import TAgentConfig

from pydantic import BaseModel, Field


# Define structured outputs
class SentimentAnalysis(BaseModel):
    score: float = Field(description="Sentiment score from 0-10")
    category: str = Field(description="positive, negative, or neutral")

class EmailDraft(BaseModel):
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")

# Build a customer feedback pipeline
pipeline = PipelineBuilder(
    name="customer_feedback_pipeline",
    description="Process customer feedback and respond appropriately"
).step(
    name="analyze_sentiment",
    goal="Analyze the sentiment of customer feedback",
    output_schema=SentimentAnalysis
).step(
    name="send_thank_you",
    goal="Draft a thank you email for positive feedback",
    depends_on=["analyze_sentiment"],
    condition=IsGreaterThan("analyze_sentiment.score", 7.0),
    output_schema=EmailDraft
).step(
    name="escalate_complaint",
    goal="Create escalation ticket for negative feedback",
    depends_on=["analyze_sentiment"],
    condition=IsLessThan("analyze_sentiment.score", 4.0)
).build()

executor_config = PipelineExecutorConfig(
    max_concurrent_steps=3,
    enable_persistence=False
)

config = TAgentConfig(model="gpt-4o-mini")

executor = PipelineExecutor(pipeline, config, executor_config)
result = await executor.execute()
```

---

## Installation

Get started in seconds. Install the core agent, or include optional extras like the Pipeline engine.

```bash
# Install the core agent
pip install tagent

# Install with the Pipeline engine
pip install "tagent[pipeline]"

# Install everything
pip install "tagent[all]"
```

---

## Dive Deeper

This README is just a glimpse of what TAgent can do. For detailed guides, tutorials, and API references, please visit our full documentation.

# ➡️ [Read the Full TAgent Documentation](./documentation/README.md)

Our documentation covers:
-   **Core Agent**: Getting started, creating tools, and configuration.
-   **Pipelines**: Building complex, multi-step workflows with dependencies, parallel execution, and advanced data flow features like `read_data`.
-   **API References** and more.

---

## Contributing

We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
# 🤖 TAgent - Modular AI Agent Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Build powerful AI agents with modular tools and automatic discovery**

TAgent is a production-ready framework for creating AI agents with modular, reusable tools. It features automatic tool discovery, dynamic schema loading, and a powerful CLI for rapid development and deployment.

**Built on [LiteLLM](https://github.com/BerriAI/litellm)** - Universal LLM API for seamless integration with 100+ language models including OpenAI, Anthropic, Azure, Google, and local models.

## ✨ Key Features

- 🔍 **Automatic Tool Discovery** - Finds and loads `tagent.tools.py` files automatically
- 🏗️ **Modular Architecture** - Reusable tools across different projects
- 📋 **Dynamic Schema Loading** - Pydantic schemas from `tagent.output.py` files
- 🚀 **Production CLI** - Professional command-line interface with console scripts
- 🔄 **Intelligent Agent Loop** - State machine-controlled adaptive planning, execution, and evaluation
- 🎯 **State Machine Control** - Prevents infinite loops and enforces logical action sequences
- 🛠️ **Rich Tool Ecosystem** - Travel planning, e-commerce, and custom tools
- 🎯 **Type-Safe Output** - Structured results with Pydantic validation
- 📝 **Comprehensive Logging** - Beautiful retro-style terminal output

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Tavernari/tagent.git
cd tagent

# One-command setup (creates venv, installs everything)
make clean-install

# Activate virtual environment
source .venv/bin/activate
```

### Basic Usage

```bash
# Test the CLI with built-in travel tools
tagent "Plan a trip from London to Rome for 2025-09-10 to 2025-09-17 with budget $2000" \
  --search-dir examples/travel_planning_cli \
  --max-iterations 10

# Quick discovery test
make cli-discovery-test

# See all available commands
make help
```

## 📁 Creating Your Own Tools

### 1. Create Tool Functions (`myproject/tagent.tools.py`)

```python
from typing import Dict, Any, Tuple, Optional

def web_search_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """Search the web for information."""
    query = args.get('query', '')
    
    # Your implementation here
    results = perform_web_search(query)
    
    return ('search_results', results)

def data_analysis_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """Analyze data and generate insights."""
    data = args.get('data', [])
    
    # Your analysis logic
    insights = analyze_data(data)
    
    return ('analysis_insights', insights)
```

### 2. Define Output Schema (`myproject/tagent.output.py`)

```python
from pydantic import BaseModel, Field
from typing import List

class ResearchReport(BaseModel):
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary") 
    findings: List[str] = Field(default=[], description="Key findings")
    recommendations: List[str] = Field(default=[], description="Actionable recommendations")
    confidence_score: float = Field(..., description="Confidence in results (0-1)")

# Required variable name
output_schema = ResearchReport
```

### 3. Run Your Agent

```bash
tagent "Research the latest trends in AI and create a comprehensive report" \
  --search-dir myproject \
  --model openrouter/ollama/gemma3 \
  --verbose
```

## 🎯 Examples

### Travel Planning Agent
```bash
# Complete travel itinerary with flights, hotels, activities
tagent "Plan a 7-day trip to Tokyo with cultural activities and budget $3000" \
  --search-dir examples/travel_planning_cli
```

### E-commerce Analysis  
```bash
# Business intelligence and product recommendations
tagent "Analyze customer data and suggest product improvements" \
  --search-dir examples/ecommerce \
  --model openrouter/ollama/gemma3
```

### Custom Research
```bash
# Multi-source research with structured output
tagent "Research sustainable energy trends and create executive summary" \
  --tools ./research/tagent.tools.py \
  --output ./research/tagent.output.py
```

## 🛠️ CLI Commands

### Development Commands
```bash
make help                  # Show all commands
make install              # Install in virtual environment  
make clean-install        # Clean install (recommended)
make test                 # Run test suite
make lint                 # Code quality checks
make format               # Format code with black
```

### CLI Commands
```bash
make cli-help             # Show CLI help
make cli-discovery-test   # Test tool discovery
make cli-test             # Full travel example
make cli-demo             # Generate demo GIF

# Direct CLI usage (after source .venv/bin/activate)
tagent --help
tagent "your goal" --search-dir path/to/tools
```

## 🏗️ Architecture

```
TAgent Framework
├── 🔍 Tool Discovery Engine
│   ├── Automatic file scanning  
│   ├── Function signature validation
│   └── Dynamic imports
├── 🤖 Intelligent Agent Core
│   ├── LLM-powered decision making (via LiteLLM)
│   ├── Adaptive planning
│   ├── Tool execution
│   └── Goal evaluation
├── 📋 Schema Management
│   ├── Pydantic integration
│   ├── Type-safe outputs
│   └── Validation
└── 🚀 Production CLI
    ├── Console scripts
    ├── Rich terminal UI
    └── Error handling
```

### 🔧 Core Dependencies

- **[LiteLLM](https://github.com/BerriAI/litellm)** - Universal LLM integration supporting 100+ models
- **Pydantic** - Type-safe data validation and schemas
- **Rich** - Beautiful terminal UI and progress indicators

## 📊 Built-in Tools

### Travel Planning (`examples/travel_planning_cli/`)
- ✈️ Flight search with budget filtering
- 🏨 Hotel recommendations with preferences
- 🎯 Activity suggestions by interest
- 💰 Cost calculation and budgeting
- 📝 Itinerary generation

### E-commerce (`examples/ecommerce/`)
- 📈 Sales analysis and trends
- 👥 Customer behavior insights
- 🛍️ Product recommendations
- 💡 Business intelligence

## 🔧 Advanced Usage

### Custom Models
TAgent provides a sophisticated model configuration system, allowing you to specify different Large Language Models (LLMs) for various stages of the agent's operation. This offers fine-grained control and optimization, as different tasks (planning, execution, summarization, evaluation, finalization) might benefit from different models (e.g., a faster, cheaper model for simple execution steps, and a more capable one for complex planning).

**Key Concepts:**

-   **Agent Steps:** TAgent's internal lifecycle is broken down into distinct steps: `PLANNER`, `EXECUTOR`, `SUMMARIZER`, `EVALUATOR`, and `FINALIZER`.
-   **Step-Specific Models:** You can assign a particular LLM to each of these steps.
-   **Global Fallback Model:** A default model that will be used for any step that doesn't have a specific model assigned.
-   **Configuration Priority:** The system follows a clear hierarchy to determine which model to use for a given step, from highest to lowest precedence:
    1.  **`AgentModelConfig` Object (Programmatic):** If you pass an `AgentModelConfig` object to the agent, its step-specific models take the highest precedence, followed by its global model.
    2.  **Environment Variables:** Step-specific environment variables (e.g., `TAGENT_PLANNER_MODEL`) override the global environment variable (`TAGENT_MODEL`).
    3.  **CLI `--model` Argument:** The `--model` argument in the CLI sets the global fallback model.
    4.  **Default Model:** If no other configuration is found, a hardcoded default model (`gpt-3.5-turbo`) is used.

**How to Configure Models:**

You have several ways to configure the models, from most specific (programmatic) to least specific (default):

1.  **Using `AgentModelConfig` (Programmatic - for advanced use cases):**
    For programmatic control, you can create an `AgentModelConfig` instance and pass it to the agent's `run` method. This allows you to define specific models for each step and a global fallback.

    ```python
    from src.tagent.model_config import AgentModelConfig

    config = AgentModelConfig(
        tagent_model="gpt-4o",  # Global fallback for all steps
        tagent_planner_model="gpt-4o-mini", # Specific model for planning
        tagent_evaluator_model="gpt-4o-mini", # Specific model for evaluation
        api_key="sk-your-api-key" # Optional API key for the LLM service
    )
    # Then pass this config object to your agent's run method, e.g.:
    # agent.run(goal="...", config=config)
    ```

2.  **Using Environment Variables (Recommended for persistent setup):**
    You can set environment variables to control the models. This is ideal for setting up your preferred models across different runs without modifying code or CLI commands every time.

    *   **Global Model:**
        ```bash
        export TAGENT_MODEL="openrouter/google/gemma-7b-it"
        ```
        This model will be used for all agent steps unless a step-specific environment variable is set.

    *   **Step-Specific Models:**
        ```bash
        export TAGENT_PLANNER_MODEL="openrouter/openai/gpt-4o-mini"
        export TAGENT_EXECUTOR_MODEL="openrouter/anthropic/claude-3-haiku"
        export TAGENT_SUMMARIZER_MODEL="openrouter/google/gemma-7b-it"
        export TAGENT_EVALUATOR_MODEL="openrouter/openai/gpt-4o-mini"
        export TAGENT_FINALIZER_MODEL="openrouter/google/gemma-7b-it"
        ```
        These will override `TAGENT_MODEL` for their respective steps.

3.  **Using the CLI `--model` Argument (Quick and easy):**
    The existing `--model` argument in the CLI now sets the global fallback model. This is the simplest way to quickly change the model for a single run.

    ```bash
    tagent "your goal" --model openrouter/google/gemma-7b-it
    ```
    If you also have environment variables set, the environment variables will take precedence over the `--model` argument.

This new system provides maximum flexibility, allowing users to tailor the LLM usage to their specific needs, optimizing for cost, speed, or capability at each stage of the agent's operation.

For a complete list of supported models and integrations, please visit the [LiteLLM Provider Documentation](https://docs.litellm.ai/docs/providers).

### Multiple Tool Directories
```bash
# Search multiple directories
tagent "complex task" \
  --search-dir ./core-tools \
  --search-dir ./specialized-tools \
  --search-dir ./custom-tools \
  --recursive
```

### API Configuration

```bash
# Set API key
export OPENAI_API_KEY="your-key"
# or
tagent "goal" --api-key your-key-here
```

## 🎬 Demo

Generate an animated demo:
```bash
make cli-demo  # Creates examples/tagent_cli_demo.gif
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow code style**: `make format && make lint`
4. **Add tests**: `pytest tests/`
5. **Commit changes**: `git commit -m "Add amazing feature"`
6. **Push branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**

### Development Setup
```bash
git clone https://github.com/Tavernari/tagent.git
cd tagent
make clean-install
source .venv/bin/activate
make test
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Why TAgent?

- **🔥 Rapid Development**: Create agents in minutes, not hours
- **🔧 Modular Design**: Reuse tools across projects
- **🚀 Production Ready**: Professional CLI and error handling
- **🎯 Type Safety**: Pydantic schemas ensure data integrity
- **🤖 Intelligent**: LLM-powered adaptive behavior
- **📈 Scalable**: From prototypes to production systems

## 🎉 Star History

If TAgent helps you build amazing AI agents, please give it a ⭐!

---

**Built with ❤️ for the AI community**

[🐛 Report Bug](https://github.com/Tavernari/tagent/issues) | [✨ Request Feature](https://github.com/Tavernari/tagent2/issues)
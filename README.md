# TAgent - When You're Tired of Unnecessarily Complex Agent Frameworks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.4.1-green.svg)](https://github.com/yourusername/tagent2)

> **A minimalist, Redux-inspired framework for AI agents that actually makes sense**

Fed up with bloated frameworks that need 50 dependencies and 200 lines of boilerplate just to make a simple automation? TAgent is a straightforward, less verbose approach to building AI agents that solve specific problems without the unnecessary complexity.

![gif](https://vhs.charm.sh/vhs-dujKmiVTP09yg9gOXAbs5.gif)

## Why TAgent?

TAgent follows a simple philosophy: **state-controlled execution with LLM fallbacks**. Instead of complex function calling or massive dependency trees, you get:

- **Redux-inspired Architecture**: Predictable state management with centralized store
- **State Machine Control**: Prevents infinite loops and unpredictable behavior  
- **Structured Outputs**: Works with any LLM via JSON, not function calling
- **Intelligent Fallbacks**: When tools don't exist, uses LLM knowledge directly
- **Zero Boilerplate**: Get started with 3 lines of code

## Quick Start

```bash
pip install -e .
```

```python
from tagent import run_agent

# That's literally all you need to start
result = run_agent(
    goal="Translate 'Hello world' to Chinese",
    model="gpt-4o-mini",
    max_iterations=3
)

print(result.get("raw_data", {}).get("llm_direct_response"))
# Output: 你好世界
```

## How It Works Under the Hood

### Deterministic State Machine
Instead of letting the LLM do whatever it wants, the agent follows a controlled flow:

```
INITIAL → PLAN → EXECUTE → EVALUATE → (loop until goal achieved)
```

Each transition is validated, preventing infinite loops and unpredictable behaviors.

### Structured Outputs Over Function Calling
No function calling dependency. The LLM returns structured JSON validated with Pydantic:

```json
{
  "action": "execute",
  "params": {"tool": "search", "args": {"query": "python"}},
  "reasoning": "Need to search for Python information"
}
```

### Intelligent Fallback System
If a tool doesn't exist, the agent uses the LLM's own knowledge as fallback. No crashes, no errors - it just works.

```python
# Tool not found? No problem!
# Agent automatically uses LLM knowledge instead
```

## Real-World Example

Here's an agent that extracts and translates TabNews articles:

```python
def extract_tabnews_articles(state, args):
    """Extract recent articles from TabNews RSS"""
    response = requests.get("https://www.tabnews.com.br/recentes/rss")
    root = ET.fromstring(response.content)
    
    articles = []
    for item in root.findall('.//item'):
        articles.append({
            "url": item.find('link').text,
            "title": item.find('title').text,
            "publication_date": item.find('pubDate').text
        })
    
    return ("articles", articles)

def translate(state, args):
    """Translate text using direct LLM call"""
    text = args.get("text", "")
    target = args.get("target_language", "")
    
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": f"Translate to {target}: {text}"}
        ]
    )
    
    return ("translation", {"translation": response.choices[0].message.content})

# Run the agent
result = run_agent(
    goal="Get 1 TabNews article, load content, summarize and translate to Chinese",
    model="gpt-4o-mini",
    tools={
        "extract_tabnews_articles": extract_tabnews_articles,
        "load_url_content": load_url_content,
        "translate": translate
    },
    max_iterations=15
)
```

The agent plans, executes tools in the correct order, and delivers structured results.

## Tool Ecosystem & Extensibility

Currently no default tools (keeping it minimal), but adapters are being developed for:
- **CrewAI** tools
- **LangChain** tools  
- **Model Context Protocol (MCP)** tools

The idea is to leverage existing ecosystems without being locked into them.

## Why Redux for Agents?

- **Predictable State**: Always know what's happening
- **Debug Friendly**: Every step is logged and inspectable
- **Composition**: Tools are pure functions, easy to test
- **Extensible**: Adding new actions is trivial
- **Time Travel**: Replay actions for debugging

## Performance & Model Support

Works with any model via LiteLLM:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Ollama (local models)
- OpenRouter
- Google Gemini
- Azure OpenAI
- And 100+ more...

Each action type can use a different model (planning with GPT-4, execution with cheaper model).

## Retro Terminal Experience

Because life's too short for boring UIs:

```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓                 T-AGENT v0.4.1 STARTING                  ▓ 
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
[-] [12:34:56] INIT: Goal: Translate hello world to Chinese
[#] [12:34:57] PLAN: Generating strategic plan...
[>] [12:34:58] EXECUTE: Using LLM fallback for translation
[+] [12:34:59] SUCCESS: Translation completed!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★                     MISSION COMPLETE                     ★ 
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

## Architecture Overview

```
TAgent Framework
├── 🎯 State Machine Controller
│   ├── Deterministic action flow
│   ├── Loop prevention
│   └── Transition validation
├── 🤖 Agent Core
│   ├── Redux-inspired store
│   ├── LLM decision making
│   ├── Tool execution
│   └── Intelligent fallbacks
├── 🛠️ Tool System
│   ├── Pure function interface
│   ├── Dynamic discovery
│   └── Type-safe signatures
└── 📊 Structured Outputs
    ├── Pydantic validation
    ├── JSON schema enforcement
    └── Type-safe results
```

## Advanced Configuration

### Model Configuration
```python
from tagent.model_config import AgentModelConfig

config = AgentModelConfig(
    tagent_model="gpt-4o",  # Global fallback
    tagent_planner_model="gpt-4o-mini",  # Planning tasks
    tagent_executor_model="gpt-3.5-turbo",  # Tool execution
    api_key="your-api-key"
)

result = run_agent(
    goal="Complex multi-step task",
    model=config,  # Pass config instead of string
    tools=my_tools,
    max_iterations=20
)
```

### Environment Variables
```bash
export TAGENT_MODEL="gpt-4o-mini"
export TAGENT_PLANNER_MODEL="gpt-4o"
export OPENAI_API_KEY="your-key"
```

## Try It Yourself

```bash
# Clone the repository
git clone https://github.com/yourusername/tagent2
cd tagent2

# Install in development mode
pip install -e .

# Run the TabNews example
python examples/tab_news_analyzer/tabnews_code_example.py
```

## Examples Directory

Check out the `/examples` folder for real implementations:

- **`tab_news_analyzer/`** - Extract and translate TabNews articles
- **`travel_planning/`** - Multi-step travel planning agent
- **`simple_qa/`** - Direct question answering without tools

Each example shows different patterns and use cases.

## Roadmap

- [ ] CrewAI/LangChain/MCP tool adapters
- [ ] Persistent memory system
- [ ] Default tool library (web search, file ops)
- [ ] Optional web interface
- [ ] Multi-agent orchestration
- [ ] Tool marketplace/registry

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -m "Add amazing feature"`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Conclusion

TAgent won't solve all the world's problems, but if you want to create agents without headaches and with code you can understand 6 months later, it might be worth a look.

The framework is small (<2000 lines), focused, and each component has a clear responsibility. Sometimes simple is better.

---

**Repository:** https://github.com/yourusername/tagent2  
**License:** MIT  
**Version:** 0.4.1

*If you made it this far and found it interesting, leave a star on GitHub. If you didn't like it, open an issue and complain - feedback is always welcome.*
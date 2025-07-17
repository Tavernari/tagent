# 1. Introduction to TAgent

## What is TAgent?

TAgent is a developer-first framework for creating powerful, task-oriented AI agents. Its core philosophy is to minimize boilerplate and adapt to your code, not the other way around.

With TAgent, you write standard Python functions, and the framework's intelligent `ToolExecutor` handles the complex parts of making them available to a Large Language Model (LLM).

## Core Philosophy

- **Task-Oriented**: Agents operate on a clear, predictable state machine (`Plan` → `Execute` → `Evaluate`) to achieve a specific `goal`. This makes their behavior easier to understand and debug.
- **Developer-First**: You don't need to learn complex abstractions. If you can write a Python function, you can create a tool for TAgent.
- **Model Agnostic**: TAgent uses structured JSON to communicate with LLMs, not model-specific "function calling" features. This ensures it works with virtually any text-generating model, from providers like Google, OpenAI, Anthropic, and more.

## When to Use a Single Agent

A single, task-based agent is the perfect tool for any objective that can be accomplished in a single, continuous process. Examples include:

- "Summarize this article."
- "Translate this text into French and German."
- "Search for the current weather in London and provide a recommendation on what to wear."
- "Write a Python script to parse a CSV file and extract all email addresses."

If your task involves multiple, dependent stages (e.g., "first do A, then do B"), you should consider using the more advanced **TAgent Pipelines**.

---

**➡️ [Next: Getting Started](./02_getting_started.md)**

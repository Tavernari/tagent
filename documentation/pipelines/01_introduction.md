# 1. Introduction to TAgent Pipelines

## What is a TAgent Pipeline?

At its core, a **TAgent Pipeline** is an automated workflow that connects multiple AI agents together to solve a complex problem.

Think of a standard TAgent as a single, skilled worker who can achieve a specific goal. A Pipeline, on the other hand, is like an entire assembly line, where each station is a specialized TAgent worker. The output of one worker becomes the input for the next, and they work in a coordinated fashion to produce a final, sophisticated result.

Technically, a pipeline is a **Directed Acyclic Graph (DAG)**, where each node in the graph is a `PipelineStep`. Each step is a self-contained TAgent with its own goal, tools, and configuration.

## Why Use a Pipeline?

While a single agent is great for straightforward tasks, pipelines excel at handling complexity and dependencies. You should consider using a pipeline when your problem involves:

- **Multiple, Dependent Stages**: For example, "first research a topic, *then* write a draft, *and then* review the draft."
- **Decomposition**: Breaking down a very large, complex goal (e.g., "build a simple website") into smaller, manageable steps.
- **Parallelism**: Performing multiple independent tasks at the same time to save time. For instance, summarizing three different articles simultaneously before combining the summaries.
- **Modularity and Reusability**: Defining clear steps makes your automation easier to debug, maintain, and modify.

## High-Level Architecture

The Pipeline engine consists of a few key components:

- **Models**: The data structures that define a `Pipeline` and its `PipelineSteps`.
- **API (`PipelineBuilder`)**: A fluent, programmatic interface for defining the structure and dependencies of your pipeline.
- **Executor (`PipelineExecutor`)**: The engine responsible for running the pipeline. It respects the dependency graph, manages the state of each step, and orchestrates the flow of data.
- **State & Memory**: A system that holds the results of completed steps and makes them available to subsequent steps.

---

Next, let's build our first pipeline.

**➡️ [Next: Getting Started: Your First Pipeline](./02_getting_started.md)**

#!/usr/bin/env python3
"""
TAgent CLI - Dynamic Tool and Schema Discovery
Discovers and loads tagent.tools.py and tagent.output.py files dynamically.
"""

import os
import sys
import glob
import importlib.util
import inspect
import argparse
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Type, List
from pydantic import BaseModel

# Import tagent from current package
from tagent.agent import run_agent


def discover_tagent_files(
    search_paths: List[str], recursive: bool = True
) -> Dict[str, List[str]]:
    """
    Discovers tagent.tools.py and tagent.output.py files in specified paths.

    Args:
        search_paths: List of directories or files to search
        recursive: Whether to search recursively

    Returns:
        Dictionary with 'tools' and 'output' keys containing file paths
    """
    discovered = {"tools": [], "output": []}

    for search_path in search_paths:
        path = Path(search_path)

        if path.is_file():
            # Direct file path provided
            if path.name == "tagent.tools.py":
                discovered["tools"].append(str(path))
            elif path.name == "tagent.output.py":
                discovered["output"].append(str(path))
        elif path.is_dir():
            # Directory - search for tagent files
            if recursive:
                pattern = "**/tagent.tools.py"
                output_pattern = "**/tagent.output.py"
            else:
                pattern = "tagent.tools.py"
                output_pattern = "tagent.output.py"

            # Find tool files
            for tools_file in path.glob(pattern):
                discovered["tools"].append(str(tools_file))

            # Find output files
            for output_file in path.glob(output_pattern):
                discovered["output"].append(str(output_file))

    return discovered


def load_tools_from_file(file_path: str) -> Dict[str, Callable]:
    """
    Loads tool functions from a tagent.tools.py file.

    Args:
        file_path: Path to the tagent.tools.py file

    Returns:
        Dictionary of tool name -> function
    """
    tools = {}

    try:
        # Load module from file path
        spec = importlib.util.spec_from_file_location("tagent_tools", file_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not load module from {file_path}")
            return tools

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find all functions in the module
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # Skip private functions
            if not name.startswith("_"):
                # Check if function has correct signature (state, args) -> result
                sig = inspect.signature(obj)
                params = list(sig.parameters.keys())

                if len(params) >= 2 and params[0] == "state" and params[1] == "args":
                    tools[name] = obj
                    print(f"  ✓ Loaded tool: {name}")
                else:
                    print(
                        f"  ⚠ Skipped {name}: Invalid signature (expected: state, args)"
                    )

    except Exception as e:
        print(f"Error loading tools from {file_path}: {e}")

    return tools


def load_output_schema_from_file(file_path: str) -> Optional[Type[BaseModel]]:
    """
    Loads output schema from a tagent.output.py file.

    Args:
        file_path: Path to the tagent.output.py file

    Returns:
        Pydantic BaseModel class or None
    """
    try:
        # Load module from file path
        spec = importlib.util.spec_from_file_location("tagent_output", file_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not load module from {file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for output_schema variable
        if hasattr(module, "output_schema"):
            schema = getattr(module, "output_schema")
            if inspect.isclass(schema) and issubclass(schema, BaseModel):
                print(f"  ✓ Loaded output schema: {schema.__name__}")
                return schema
            else:
                print(f"  ⚠ output_schema is not a Pydantic BaseModel class")
        else:
            print(f"  ⚠ No output_schema variable found in {file_path}")

    except Exception as e:
        print(f"Error loading output schema from {file_path}: {e}")

    return None


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="TAgent CLI - Run agents with dynamic tool discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tagent "Plan a trip to Rome" --tools ./travel/tagent.tools.py
  python -m tagent "Find the best products" --search-dir ./ecommerce --recursive
  python -m tagent "Generate report" --search-dir . --model openrouter/gpt-4
  python -m tagent "Simple task" --model openrouter/gpt-4  # Run without tools
        """,
    )

    parser.add_argument("goal", help="The goal for the agent to achieve")

    # Tool discovery options
    parser.add_argument(
        "--tools",
        action="append",
        default=[],
        help="Specific tagent.tools.py file paths (can be used multiple times)",
    )
    parser.add_argument("--output", help="Specific tagent.output.py file path")
    parser.add_argument(
        "--search-dir",
        action="append",
        default=[],
        help="Directories to search for tagent files (can be used multiple times). If no paths are provided, agent runs without tools.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search directories recursively (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive search",
    )

    # Agent configuration
    parser.add_argument(
        "--model",
        default="openrouter/google/gemini-2.5-pro",
        help="LLM model to use (default: openrouter/google/gemini-2.5-pro)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of iterations (default: 20)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--api-key", help="API key for the LLM service (or set OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()

    print("🤖 TAgent CLI - Dynamic Tool Discovery")
    print("=" * 50)

    # Determine search paths
    search_paths = []

    # Add explicit tool files
    if args.tools:
        search_paths.extend(args.tools)

    # Add explicit output file
    if args.output:
        search_paths.append(args.output)

    # Add search directories
    if args.search_dir:
        search_paths.extend(args.search_dir)

    # Default to no tools if no paths specified
    if not search_paths:
        print("No paths specified, running with no tools...")

    print(f"Search paths: {search_paths}")
    print(f"Recursive search: {args.recursive}")
    print()

    # Discover tagent files
    print("🔍 Discovering tagent files...")
    discovered = discover_tagent_files(search_paths, args.recursive) if search_paths else {"tools": [], "output": []}

    print(f"Found {len(discovered['tools'])} tool files:")
    for tool_file in discovered["tools"]:
        print(f"  📁 {tool_file}")

    print(f"Found {len(discovered['output'])} output files:")
    for output_file in discovered["output"]:
        print(f"  📄 {output_file}")
    print()

    # Load tools
    print("🔧 Loading tools...")
    all_tools = {}

    for tool_file in discovered["tools"]:
        print(f"Loading from {tool_file}:")
        tools = load_tools_from_file(tool_file)
        all_tools.update(tools)

    print(f"Total tools loaded: {len(all_tools)}")
    if all_tools:
        print("Available tools:", list(all_tools.keys()))
    print()

    # Load output schema
    output_schema = None
    if discovered["output"]:
        print("📋 Loading output schema...")
        # Use the first output file found
        output_file = discovered["output"][0]
        if len(discovered["output"]) > 1:
            print(f"Multiple output files found, using: {output_file}")

        print(f"Loading from {output_file}:")
        output_schema = load_output_schema_from_file(output_file)
        print()

    # Check if we have everything needed
    if not all_tools:
        print("⚠️  Warning: No tools found. Agent will run with limited capabilities.")

    # Run the agent
    print("🚀 Starting agent...")
    print(f"Goal: {args.goal}")
    print(f"Model: {args.model}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Verbose: {args.verbose}")
    print("=" * 50)
    print()

    try:
        result = run_agent(
            goal=args.goal,
            model=args.model,
            api_key=args.api_key,
            max_iterations=args.max_iterations,
            tools=all_tools if all_tools else None,
            output_format=output_schema,
            verbose=args.verbose,
        )

        print("\n" + "=" * 50)
        print("🎯 Agent execution completed!")

        if result:
            print("\n📊 Final Result:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if key == "conversation_history":
                        print(f"{key}: {len(value)} messages")
                    elif key == "chat_summary":
                        continue  # Skip chat summary in main output
                    else:
                        print(f"{key}: {value}")
            else:
                print(result)
        else:
            print("No result returned.")

    except KeyboardInterrupt:
        print("\n⚠️  Agent execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during agent execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

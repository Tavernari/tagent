#!/usr/bin/env python3
"""
Travel Planning Agent CLI Example

This example shows how to use the TAgent CLI with separate tagent.tools.py and tagent.output.py files.
Instead of calling run_agent directly, this demonstrates using the main.py CLI interface.

Usage:
    python examples/travel_planning_cli_example.py

This will internally call the main.py CLI with the appropriate parameters.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_travel_agent_cli():
    """Run the travel planning agent using the main.py CLI."""
    
    # Define travel parameters
    goal = (
        "I need help planning a realistic trip from London to Rome for the dates 2025-09-10 to 2025-09-17, "
        "staying within a total budget of about $2000.00. Please start by searching for affordable flights "
        "that fit the dates and budget. If no flights are available, note that and suggest alternatives. "
        "Then, find hotel options in Rome that match the dates and remaining budget. "
        "After that, recommend activities focused on food that can be done during the trip. "
        "Finally, calculate the total cost and provide a detailed itinerary summary."
    )
    
    # Path to the CLI tools and output schema
    tools_dir = Path(__file__).parent / "travel_planning_cli"
    
    # Build the CLI command
    cli_command = [
        sys.executable,  # Use the same Python interpreter
        "main.py",
        goal,
        "--search-dir", str(tools_dir),
        "--model", "openrouter/google/gemini-2.5-pro",
        "--max-iterations", "15",
        "--verbose"
    ]
    
    print("🚀 Starting Travel Planning Agent via CLI")
    print("=" * 60)
    print(f"Goal: {goal[:100]}...")
    print(f"Tools directory: {tools_dir}")
    print(f"CLI command: {' '.join(cli_command[:3])} [goal] [options]")
    print("=" * 60)
    print()
    
    try:
        # Run the CLI command
        result = subprocess.run(
            cli_command,
            cwd=Path(__file__).parent.parent,  # Run from project root
            capture_output=False,  # Let output stream directly to console
            text=True,
            check=True
        )
        
        print("\n" + "=" * 60)
        print("✅ Travel planning completed successfully!")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ CLI execution failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

def show_cli_alternatives():
    """Show alternative ways to call the CLI directly."""
    
    tools_dir = Path(__file__).parent / "travel_planning_cli"
    
    print("\n" + "=" * 60)
    print("📖 Alternative CLI Usage Examples")
    print("=" * 60)
    
    print("\n1. Direct CLI call (from project root):")
    print('python main.py "Plan a trip to Rome from London" \\')
    print(f'  --search-dir {tools_dir} \\')
    print('  --model openrouter/google/gemini-2.5-pro \\')
    print('  --verbose')
    
    print("\n2. Specify exact files:")
    print('python main.py "Plan a trip to Rome" \\')
    print(f'  --tools {tools_dir}/tagent.tools.py \\')
    print(f'  --output {tools_dir}/tagent.output.py \\')
    print('  --model openrouter/google/gemini-2.5-pro')
    
    print("\n3. Search multiple directories:")
    print('python main.py "Plan a trip" \\')
    print(f'  --search-dir {tools_dir} \\')
    print('  --search-dir ./examples/travel \\')
    print('  --recursive')
    
    print("\n4. Quick test with minimal options:")
    print('python main.py "Plan a trip to Rome" \\')
    print(f'  --search-dir {tools_dir}')
    
    print("\n💡 Tips:")
    print("- Use --verbose for detailed execution logs")
    print("- Use --no-recursive to disable recursive search")
    print("- Set OPENAI_API_KEY environment variable or use --api-key")
    print("- Check available models with your LLM provider")

if __name__ == "__main__":
    print("🤖 TAgent CLI Travel Planning Example")
    print("This example demonstrates using main.py CLI with modular tagent files.")
    print()
    
    # Check if we're in the right directory
    if not (Path.cwd() / "main.py").exists():
        print("❌ Please run this script from the project root directory (where main.py is located)")
        print(f"Current directory: {Path.cwd()}")
        print("Try: cd /path/to/tagent2 && python examples/travel_planning_cli_example.py")
        sys.exit(1)
    
    # Check if tools directory exists
    tools_dir = Path(__file__).parent / "travel_planning_cli"
    if not tools_dir.exists():
        print(f"❌ Tools directory not found: {tools_dir}")
        print("Make sure the travel_planning_cli directory with tagent.tools.py and tagent.output.py exists")
        sys.exit(1)
    
    # Run the agent
    try:
        exit_code = run_travel_agent_cli()
        
        # Show additional examples
        show_cli_alternatives()
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        sys.exit(1)
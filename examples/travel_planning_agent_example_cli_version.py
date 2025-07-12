#!/usr/bin/env python3
"""
Travel Planning Agent - CLI Version

This is an alternative version of travel_planning_agent_example.py that uses the main.py CLI
instead of calling run_agent directly. This demonstrates two different approaches:

1. Original: Direct use of run_agent with inline tool definitions
2. CLI Version: Using modular tagent.tools.py and tagent.output.py with main.py CLI

Both approaches achieve the same result but with different levels of modularity and reusability.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_via_direct_import():
    """Run using the original direct import approach (for comparison)."""
    print("🔧 Method 1: Direct run_agent import (Original Approach)")
    print("-" * 50)
    
    # This would be the original approach - importing and calling run_agent directly
    # We'll skip the actual execution here to focus on the CLI comparison
    print("This would import run_agent directly and call it with inline tool definitions.")
    print("Tools are defined in the same file as functions.")
    print("Less modular but simpler for single-use cases.")
    print()

def run_via_cli():
    """Run using the new CLI approach."""
    print("🚀 Method 2: CLI with Modular Tools (New Approach)")
    print("-" * 50)
    
    # Define the goal
    goal = (
        "I need help planning a realistic trip from London to Rome for the dates 2025-09-10 to 2025-09-17, "
        "staying within a total budget of about $2000.00. Please start by searching for affordable flights "
        "that fit the dates and budget. Then, find hotel options in Rome that match the dates and remaining budget. "
        "After that, recommend activities focused on food that can be done during the trip. "
        "Finally, calculate the total cost and provide a detailed itinerary summary."
    )
    
    # Path to the modular tools
    tools_dir = Path(__file__).parent / "travel_planning_cli"
    
    if not tools_dir.exists():
        print(f"❌ Tools directory not found: {tools_dir}")
        print("Run the travel_planning_cli_example.py first to see this in action.")
        return False
    
    # Build CLI command
    cli_command = [
        sys.executable,
        "main.py",
        goal,
        "--search-dir", str(tools_dir),
        "--model", "openrouter/google/gemini-2.5-pro",
        "--max-iterations", "12",
        # Remove --verbose for cleaner output in comparison
    ]
    
    print(f"Executing CLI command from tools in: {tools_dir}")
    print(f"Command: python main.py \"[goal]\" --search-dir {tools_dir.name} [options]")
    print()
    
    try:
        # Execute the CLI
        result = subprocess.run(
            cli_command,
            cwd=Path(__file__).parent.parent,  # Project root
            capture_output=True,  # Capture for cleaner comparison output
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ CLI execution completed successfully!")
            print("\n📊 Output Summary:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"❌ CLI execution failed (return code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ CLI execution timed out")
        return False
    except Exception as e:
        print(f"❌ Error executing CLI: {e}")
        return False

def compare_approaches():
    """Compare the two approaches."""
    print("\n" + "=" * 60)
    print("📋 COMPARISON: Direct Import vs CLI Approach")
    print("=" * 60)
    
    comparison = {
        "Aspect": ["Modularity", "Reusability", "Setup Complexity", "Tool Sharing", "Debugging", "Production Ready"],
        "Direct Import": [
            "Low - tools in same file",
            "Low - tools tied to script", 
            "Simple - single file",
            "Hard - copy/paste needed",
            "Easy - direct debugging",
            "Basic - good for prototypes"
        ],
        "CLI Approach": [
            "High - separate tagent files",
            "High - tools easily shared",
            "Medium - multiple files", 
            "Easy - tagent files discoverable",
            "Medium - CLI layer",
            "High - scalable architecture"
        ]
    }
    
    # Print comparison table
    print(f"{'Aspect':<20} {'Direct Import':<25} {'CLI Approach'}")
    print("-" * 70)
    
    for i, aspect in enumerate(comparison["Aspect"]):
        direct = comparison["Direct Import"][i]
        cli = comparison["CLI Approach"][i]
        print(f"{aspect:<20} {direct:<25} {cli}")
    
    print("\n💡 Recommendations:")
    print("- Use Direct Import for: Quick prototypes, single-use scripts, learning")
    print("- Use CLI Approach for: Production systems, reusable tools, team projects")
    
def show_file_structure():
    """Show the file structure difference."""
    print("\n" + "=" * 60)
    print("📁 FILE STRUCTURE COMPARISON")
    print("=" * 60)
    
    print("\n🔧 Direct Import Approach:")
    print("examples/")
    print("├── travel_planning_agent_example.py    # All-in-one file")
    print("│   ├── Tool functions defined inline")
    print("│   ├── Pydantic models defined inline") 
    print("│   └── run_agent() called directly")
    
    print("\n🚀 CLI Approach:")
    print("examples/")
    print("├── travel_planning_cli/")
    print("│   ├── tagent.tools.py     # Reusable tool functions")
    print("│   └── tagent.output.py    # Reusable output schema")
    print("├── travel_planning_cli_example.py  # CLI orchestration")
    print("└── main.py (root)          # CLI discovery and execution")
    
    print("\n✨ Benefits of CLI Structure:")
    print("- Tools can be discovered automatically")
    print("- Easy to share tools between projects")
    print("- Clear separation of concerns")
    print("- Can mix and match different tool sets")
    print("- Production-ready scaling")

if __name__ == "__main__":
    print("🤖 TAgent: Comparing Direct Import vs CLI Approaches")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not (Path.cwd() / "main.py").exists():
        print("❌ Please run this script from the project root directory")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)
    
    # Show file structure first
    show_file_structure()
    
    print("\n" + "=" * 60)
    print("🔄 EXECUTION COMPARISON")
    print("=" * 60)
    
    # Compare approaches
    run_via_direct_import()
    
    print("\n" + "-" * 60)
    
    # Run CLI version
    success = run_via_cli()
    
    # Show comparison
    compare_approaches()
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION")
    print("=" * 60)
    
    if success:
        print("✅ Both approaches work! Choose based on your needs:")
        print("   - Direct Import: Simple, fast prototyping")
        print("   - CLI Approach: Scalable, reusable, production-ready")
    else:
        print("ℹ️  CLI approach shown conceptually (tools may need API keys)")
        print("   Both approaches are valid - choose based on your requirements")
    
    print("\n🚀 Next Steps:")
    print("1. Try: python examples/travel_planning_cli_example.py")
    print("2. Create your own tagent.tools.py files")
    print("3. Use main.py CLI for production workflows")
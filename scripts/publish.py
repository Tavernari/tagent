#!/usr/bin/env python3
"""
TAgent Package Publishing Script

This script automates the process of building and publishing TAgent to PyPI.
It handles both testing (TestPyPI) and production (PyPI) environments.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True, cwd=None):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Exit code: {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning previous build artifacts...")
    
    # Remove build directories
    dirs_to_remove = ["build", "dist", "src/tagent.egg-info"]
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            run_command(f"rm -rf {dir_name}")
    
    print("✅ Build artifacts cleaned")


def run_tests():
    """Run the test suite."""
    print("🧪 Running test suite...")
    
    result = run_command("python -m pytest tests/ -v", check=False)
    if result.returncode != 0:
        print("❌ Tests failed!")
        return False
    
    print("✅ Tests passed")
    return True


def run_linting():
    """Run code quality checks."""
    print("🔍 Running code quality checks...")
    
    # Run black check
    result = run_command("python -m black --check src/", check=False)
    if result.returncode != 0:
        print("❌ Black formatting check failed!")
        print("Run: python -m black src/ to fix formatting")
        return False
    
    # Run flake8
    result = run_command("python -m flake8 src/", check=False)
    if result.returncode != 0:
        print("❌ Flake8 linting failed!")
        return False
    
    print("✅ Code quality checks passed")
    return True


def build_package():
    """Build the package distributions."""
    print("📦 Building package distributions...")
    
    # Install build dependencies
    run_command("python -m pip install --upgrade build twine")
    
    # Build the package
    run_command("python -m build")
    
    # Check the distributions
    run_command("python -m twine check dist/*")
    
    print("✅ Package built successfully")


def publish_to_testpypi():
    """Publish to TestPyPI for testing."""
    print("🚀 Publishing to TestPyPI...")
    
    run_command("python -m twine upload --repository testpypi dist/*")
    
    print("✅ Published to TestPyPI")
    print("📝 Test installation with:")
    print("   pip install --index-url https://test.pypi.org/simple/ tagent")


def publish_to_pypi():
    """Publish to production PyPI."""
    print("🚀 Publishing to PyPI...")
    
    # Final confirmation
    response = input("⚠️  You're about to publish to production PyPI. Continue? (yes/no): ")
    if response.lower() != "yes":
        print("❌ Publication cancelled")
        return
    
    run_command("python -m twine upload dist/*")
    
    print("✅ Published to PyPI")
    print("📝 Install with: pip install tagent")


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(description="TAgent package publishing script")
    parser.add_argument(
        "--target",
        choices=["test", "prod"],
        default="test",
        help="Publishing target: test (TestPyPI) or prod (PyPI)"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--skip-lint",
        action="store_true",
        help="Skip linting checks"
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean build artifacts"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🏗️  TAgent Package Publishing Script")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🎯 Target: {'TestPyPI' if args.target == 'test' else 'PyPI'}")
    print()
    
    # Clean build artifacts
    clean_build()
    
    if args.clean_only:
        print("✅ Clean complete")
        return
    
    # Run quality checks
    if not args.skip_tests:
        if not run_tests():
            print("❌ Publishing cancelled due to test failures")
            sys.exit(1)
    
    if not args.skip_lint:
        if not run_linting():
            print("❌ Publishing cancelled due to linting failures")
            sys.exit(1)
    
    # Build package
    build_package()
    
    # Publish
    if args.target == "test":
        publish_to_testpypi()
    else:
        publish_to_pypi()
    
    print()
    print("🎉 Publishing complete!")


if __name__ == "__main__":
    main()
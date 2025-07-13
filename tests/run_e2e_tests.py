#!/usr/bin/env python3
"""
Test runner for TAgent end-to-end price comparison tests.

Usage:
    python tests/run_e2e_tests.py              # Run mock tests only
    python tests/run_e2e_tests.py --openrouter # Run with OpenRouter (requires API key)
    python tests/run_e2e_tests.py --all        # Run all tests
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_e2e_price_comparison import (
    test_price_comparison_e2e_with_mock_llm,
    test_price_comparison_e2e_with_openrouter
)
from test_config import get_openrouter_api_key


def run_mock_test():
    """Run the mock LLM test."""
    print("=" * 60)
    print("🧪 Running Mock LLM Test")
    print("=" * 60)
    try:
        test_price_comparison_e2e_with_mock_llm()
        print("✅ Mock test PASSED")
        return True
    except Exception as e:
        print(f"❌ Mock test FAILED: {e}")
        return False


def run_openrouter_test():
    """Run the OpenRouter test."""
    print("=" * 60)
    print("🌐 Running OpenRouter Test")
    print("=" * 60)
    
    api_key = get_openrouter_api_key()
    if not api_key:
        print("⚠️  OpenRouter test SKIPPED: No API key found")
        print("   Set OPENROUTER_API_KEY environment variable to run this test")
        return True  # Skip is not a failure
    
    try:
        test_price_comparison_e2e_with_openrouter()
        print("✅ OpenRouter test PASSED")
        return True
    except Exception as e:
        print(f"❌ OpenRouter test FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run TAgent E2E tests")
    parser.add_argument(
        "--openrouter", 
        action="store_true", 
        help="Run OpenRouter test (requires API key)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    results = []
    
    # Always run mock test unless only openrouter is specified
    if not args.openrouter or args.all:
        results.append(run_mock_test())
    
    # Run OpenRouter test if requested
    if args.openrouter or args.all:
        results.append(run_openrouter_test())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"🎉 All tests passed! ({passed}/{total})")
        return 0
    else:
        print(f"💥 Some tests failed. ({passed}/{total} passed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
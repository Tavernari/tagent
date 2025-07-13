#!/usr/bin/env python3
"""
Test script for the new step-specific model configuration system.
"""

import os
from src.tagent.model_config import AgentModelConfig, ModelConfig, AgentStep

def test_environment_variables():
    """Test model selection with environment variables."""
    print("=== Testing Environment Variables ===")
    
    # Clear any existing env vars
    env_vars = [
        "TAGENT_MODEL",
        "TAGENT_PLANNER_MODEL", 
        "TAGENT_EXECUTOR_MODEL",
        "TAGENT_SUMMARIZER_MODEL",
        "TAGENT_EVALUATOR_MODEL",
        "TAGENT_FINALIZER_MODEL"
    ]
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    
    # Test default behavior
    planner_model = ModelConfig.get_model_for_step(AgentStep.PLANNER)
    print(f"Default planner model: {planner_model}")
    assert planner_model == "gpt-3.5-turbo"
    
    # Test global env var
    os.environ["TAGENT_MODEL"] = "gpt-4"
    planner_model = ModelConfig.get_model_for_step(AgentStep.PLANNER)
    print(f"With global env: {planner_model}")
    assert planner_model == "gpt-4"
    
    # Test specific env var overrides global
    os.environ["TAGENT_PLANNER_MODEL"] = "gpt-4-turbo"
    planner_model = ModelConfig.get_model_for_step(AgentStep.PLANNER)
    executor_model = ModelConfig.get_model_for_step(AgentStep.EXECUTOR)
    print(f"Planner with specific env: {planner_model}")
    print(f"Executor with global env: {executor_model}")
    assert planner_model == "gpt-4-turbo"
    assert executor_model == "gpt-4"
    
    # Clean up
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    
    print("✓ Environment variables test passed\n")


def test_config_object():
    """Test model selection with AgentModelConfig object."""
    print("=== Testing Config Object ===")
    
    # Test basic config
    config = AgentModelConfig(
        tagent_model="claude-3",
        tagent_planner_model="gpt-4-turbo",
        api_key="test-key"
    )
    
    planner_model = ModelConfig.get_model_for_step(AgentStep.PLANNER, config=config)
    executor_model = ModelConfig.get_model_for_step(AgentStep.EXECUTOR, config=config)
    
    print(f"Planner with config: {planner_model}")
    print(f"Executor with config: {executor_model}")
    
    assert planner_model == "gpt-4-turbo"  # Specific override
    assert executor_model == "claude-3"    # Falls back to global
    assert config.api_key == "test-key"
    
    print("✓ Config object test passed\n")


def test_priority_order():
    """Test that config object takes priority over environment variables."""
    print("=== Testing Priority Order ===")
    
    # Set environment variables
    os.environ["TAGENT_MODEL"] = "env-global-model"
    os.environ["TAGENT_PLANNER_MODEL"] = "env-planner-model"
    
    # Create config object
    config = AgentModelConfig(
        tagent_model="config-global-model",
        tagent_planner_model="config-planner-model"
    )
    
    # Test that config takes priority
    planner_model = ModelConfig.get_model_for_step(AgentStep.PLANNER, config=config)
    executor_model = ModelConfig.get_model_for_step(AgentStep.EXECUTOR, config=config)
    
    print(f"Planner (config priority): {planner_model}")
    print(f"Executor (config priority): {executor_model}")
    
    assert planner_model == "config-planner-model"  # Config specific wins
    assert executor_model == "config-global-model"  # Config global wins
    
    # Test env vars still work when no config
    planner_env = ModelConfig.get_model_for_step(AgentStep.PLANNER)
    executor_env = ModelConfig.get_model_for_step(AgentStep.EXECUTOR)
    
    print(f"Planner (env only): {planner_env}")
    print(f"Executor (env only): {executor_env}")
    
    assert planner_env == "env-planner-model"
    assert executor_env == "env-global-model"
    
    # Clean up
    del os.environ["TAGENT_MODEL"]
    del os.environ["TAGENT_PLANNER_MODEL"]
    
    print("✓ Priority order test passed\n")


def test_convenience_functions():
    """Test the convenience functions work correctly."""
    print("=== Testing Convenience Functions ===")
    
    from src.tagent.model_config import (
        get_planner_model,
        get_executor_model,
        get_summarizer_model,
        get_evaluator_model,
        get_finalizer_model,
        create_config_from_string
    )
    
    # Test string to config conversion
    config = create_config_from_string("test-model", "test-api-key")
    assert config.tagent_model == "test-model"
    assert config.api_key == "test-api-key"
    assert config.tagent_planner_model is None
    
    # Test convenience functions
    config2 = AgentModelConfig(
        tagent_model="base-model",
        tagent_planner_model="special-planner"
    )
    
    planner = get_planner_model("fallback", config2)
    executor = get_executor_model("fallback", config2)
    
    print(f"Convenience planner: {planner}")
    print(f"Convenience executor: {executor}")
    
    assert planner == "special-planner"
    assert executor == "base-model"
    
    print("✓ Convenience functions test passed\n")


def main():
    """Run all tests."""
    print("Testing Step-Specific Model Configuration System\n")
    
    try:
        test_environment_variables()
        test_config_object()
        test_priority_order()
        test_convenience_functions()
        
        print("🎉 All tests passed! The step-specific model configuration system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Example demonstrating per-step TAgentConfig configuration.

This example shows how to use different TAgent configurations for different 
pipeline steps, with the main config as fallback.
"""

from src.tagent.pipeline import Pipeline, PipelineStep, PipelineExecutor
from src.tagent.config import TAgentConfig
from src.tagent.models import AgentModelConfig


async def main():
    """Demonstrate per-step configuration."""
    
    # Main pipeline configuration (used as fallback)
    main_config = TAgentConfig(
        model="gemini/gemini-pro",
        api_key="your-api-key",
        max_iterations=10,
        temperature=0.0,
        verbose=True
    )
    
    # Step-specific configuration for creative tasks
    creative_config = TAgentConfig(
        model="gemini/gemini-pro",  # Could use different model
        temperature=0.8,  # Higher temperature for creativity
        max_iterations=15,  # More iterations for complex creative tasks
        verbose=False  # Less verbose for this step
    )
    
    # Step-specific configuration for analytical tasks
    analytical_config = TAgentConfig(
        model="gemini/gemini-pro",
        temperature=0.1,  # Lower temperature for precise analysis
        max_iterations=5,  # Fewer iterations for focused analysis
        verbose=True  # More verbose for debugging
    )
    
    # Create pipeline with mixed configurations
    pipeline = Pipeline(
        name="mixed_config_pipeline",
        description="Pipeline demonstrating per-step configurations"
    )
    
    # Step 1: Uses main config (no agent_config specified)
    step1 = PipelineStep(
        name="gather_data",
        goal="Gather information about a topic",
        # agent_config=None  # Will use main_config as fallback
    )
    
    # Step 2: Uses creative config for brainstorming
    step2 = PipelineStep(
        name="brainstorm_ideas",
        goal="Generate creative ideas based on the gathered data",
        depends_on=["gather_data"],
        agent_config=creative_config  # Custom config for creativity
    )
    
    # Step 3: Uses analytical config for evaluation
    step3 = PipelineStep(
        name="analyze_ideas",
        goal="Analyze and evaluate the generated ideas",
        depends_on=["brainstorm_ideas"],
        agent_config=analytical_config  # Custom config for analysis
    )
    
    # Step 4: Back to main config for final summary
    step4 = PipelineStep(
        name="final_summary",
        goal="Create a final summary of the analysis",
        depends_on=["analyze_ideas"],
        # agent_config=None  # Will use main_config as fallback
    )
    
    # Add steps to pipeline
    pipeline.add_step(step1)
    pipeline.add_step(step2)
    pipeline.add_step(step3)
    pipeline.add_step(step4)
    
    # Create executor with main config
    executor = PipelineExecutor(pipeline, main_config)
    
    print("Pipeline Configuration Summary:")
    print(f"Main config: model={main_config.model}, temp={main_config.temperature}")
    print(f"Creative step: model={creative_config.model}, temp={creative_config.temperature}")
    print(f"Analytical step: model={analytical_config.model}, temp={analytical_config.temperature}")
    print("\nStep configuration resolution:")
    print("- gather_data: uses main config (fallback)")
    print("- brainstorm_ideas: uses creative config (custom)")
    print("- analyze_ideas: uses analytical config (custom)")
    print("- final_summary: uses main config (fallback)")
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    # result = await executor.execute()
    # print(f"Pipeline completed: {result.success}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
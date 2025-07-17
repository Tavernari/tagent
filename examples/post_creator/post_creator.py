#!/usr/bin/env python3

import asyncio
from typing import Tuple
from pydantic import BaseModel, Field
import os
from os import path

from tagent.pipeline import PipelineBuilder, ExecutionMode
from tagent.pipeline.conditions import And, DataExists
from tagent.pipeline.executor import PipelineExecutor, PipelineExecutorConfig
from tagent.config import TAgentConfig

def read_positive_text() -> Tuple[str, str]:
    """Reads the text from the positive_text.md file."""
    try:
        with open(path.join("examples", "post_creator", "text_positive.md"), "r", encoding='utf-8') as f:
            text = f.read()
            return ("positive_text", text)
    except Exception as e:
        print(f"Failed to read positive text: {str(e)}")
        return ("positive_text", "failed to read positive text")

def read_negative_text() -> Tuple[str, str]:
    """Reads the text from the negative_text.md file."""
    try:
        with open(path.join("examples", "post_creator", "text_negative.md"), "r", encoding='utf-8') as f:
            text = f.read()
            return ("negative_text", text)
    except Exception as e:
        print(f"Failed to read negative text: {str(e)}")
        return ("negative_text", "failed to read negative text")

def read_neutral_text() -> Tuple[str, str]:
    """Reads the text from the neutral_text.md file."""
    try:
        with open(path.join("examples", "post_creator", "text_neutral.md"), "r", encoding='utf-8') as f:
            text = f.read()
            return ("neutral_text", text)
    except Exception as e:
        print(f"Failed to read neutral text: {str(e)}")
        return ("neutral_text", "failed to read neutral text")

def save_post(post_content: str) -> Tuple[str, str]:
    """Saves the generated post to a file."""
    try:
        if not post_content:
            return ("post", "failed to save post: no post content provided")
        
        with open(path.join("examples", "post_creator", "final_post.md"), "w", encoding='utf-8') as f:
            f.write(post_content)
            return ("post", "post saved successfully")
    except Exception as e:
        print(f"Failed to save post: {str(e)}")
        return ("post", f"failed to save post: {str(e)}")

# Define the expected output structure for the LLM-based step
class BlogPostOutput(BaseModel):
    post: str = Field(description="The full content of the generated blog post, synthesizing the provided texts.")

class TextExtracted(BaseModel):
    text: str = Field(description="The extracted text from the file.")

async def main():
    print("\n🚀 POST CREATOR PIPELINE WITH CONCURRENT UI")
    print("=" * 50)
    
    # Create pipeline
    pipeline = PipelineBuilder(
        name="post_creator_pipeline",
        description="The final goal of this pipeline is to create a blog post.",
    ).step(
        name="positive_text_step",
        goal="Load the positive text from the file",
        execution_mode=ExecutionMode.CONCURRENT,
        tools=[
            read_positive_text,
        ],
        output_schema=TextExtracted,
    ).step(
        name="negative_text_step",
        goal="Load the negative text from the file",
        execution_mode=ExecutionMode.CONCURRENT,
        tools=[
            read_negative_text,
        ],
        output_schema=TextExtracted,
    ).step(
        name="neutral_text_step",
        goal="Load the neutral text from the file",
        execution_mode=ExecutionMode.CONCURRENT,
        tools=[
            read_neutral_text,
        ],
        output_schema=TextExtracted,
    ).step(
        name="post_creation_step",
        goal=(
            "Create a comprehensive blog post by synthesizing the 'positive_text', 'negative_text'"
            ", and 'neutral_text' from the context. The post should present a balanced view on the "
            "topic of Context Engineering."
        ),
        depends_on=["positive_text_step", "negative_text_step", "neutral_text_step"],
        condition=And(
            DataExists("positive_text_step"),
            DataExists("negative_text_step"),
            DataExists("neutral_text_step"),
        ),
        output_schema=BlogPostOutput,
        read_data=["positive_text_step.text", "negative_text_step.text", "neutral_text_step.text"],
    ).step(
        name="publish_post_step",
        goal="Publish the post by saving the content from the 'post_creation' step to a file.",
        depends_on=["post_creation_step"],
        tools=[
            save_post,
        ],
        condition=DataExists("post_creation_step"),
        read_data=["post_creation_step.post"],
    ).build()

    # Configure TAgent
    config = TAgentConfig(
        model="openrouter/google/gemini-2.5-flash-lite-preview-06-17",
        verbose=True,
    )

    # Configure executor with concurrent UI
    executor_config = PipelineExecutorConfig(
        max_concurrent_steps=3,
        enable_persistence=False
    )

    print("🎯 Starting pipeline execution with concurrent UI...")
    print("📊 Watch the progress dashboard below:")
    print()

    # Execute pipeline
    executor = PipelineExecutor(pipeline, config, executor_config)
    result = await executor.execute()

    print("\n✅ Pipeline execution completed!")
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Steps completed: {result.steps_completed}")
    print(f"Steps failed: {result.steps_failed}")

    if result.success:
        print("\n📄 Blog post created successfully!")
        if os.path.exists("examples/post_creator/final_post.md"):
            print("📁 Saved to: examples/post_creator/final_post.md")
        else:
            print("⚠️  Post file not found")

    return result

if __name__ == "__main__":
    asyncio.run(main())
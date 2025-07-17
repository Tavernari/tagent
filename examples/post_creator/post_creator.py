#!/usr/bin/env python3

import asyncio
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field
import random
import os
from os import path

from tagent.pipeline import PipelineBuilder, ConditionDSL, ExecutionMode
from tagent.pipeline.executor import PipelineExecutor, PipelineExecutorConfig
from tagent.config import TAgentConfig

def read_positive_text() -> Tuple[str, str]:
    """Reads the text from the positive_text.md file."""
    try:
        with open(path.join("examples", "post_creator", "text_positive.md"), "r") as f:
            text = f.read()
            return ("positive_text", text)
    except Exception as e:
        return ("positive_text", "failed to read positive text")

def read_negative_text() -> Tuple[str, str]:
    """Reads the text from the negative_text.md file."""
    try:
        with open(path.join("examples", "post_creator", "text_negative.md"), "r") as f:
            text = f.read()
            return ("negative_text", text)
    except Exception as e:
        return ("negative_text", "failed to read negative text")

def read_neutral_text() -> Tuple[str, str]:
    """Reads the text from the neutral_text.md file."""
    try:
        with open(path.join("examples", "post_creator", "text_neutral.md"), "r") as f:
            text = f.read()
            return ("neutral_text", text)
    except Exception as e:
        return ("neutral_text", "failed to read neutral text")

def save_post(post: str) -> Tuple[str, str]:
    """Saves the generated post to a file."""
    try:
        with open(path.join("examples", "post_creator", "final_post.md"), "w") as f:
            f.write(post)
            return ("post", "post saved successfully")
    except Exception as e:
        return ("post", "failed to save post")

# Define the expected output structure for the LLM-based step
class BlogPostOutput(BaseModel):
    post: str = Field(description="The full content of the generated blog post, synthesizing the provided texts.")

async def main():
    print("\n🚀 POST CREATOR PIPELINE WITH CONCURRENT UI")
    print("=" * 50)
    
    # Create pipeline
    pipeline = PipelineBuilder(
        name="post_creator_pipeline",
        description="The final goal of this pipeline is to create a blog post.",
    ).step(
        name="read_positive_texts",
        goal="Load the positive text from the file",
        execution_mode=ExecutionMode.CONCURRENT,
        tools_filter=["read_positive_text"],
    ).step(
        name="read_negative_texts",
        goal="Load the negative text from the file",
        execution_mode=ExecutionMode.CONCURRENT,
        tools_filter=["read_negative_text"],
    ).step(
        name="read_neutral_texts",
        goal="Load the neutral text from the file",
        execution_mode=ExecutionMode.CONCURRENT,
        tools_filter=["read_neutral_text"],
    ).step(
        name="post_creation",
        goal=(
            "Create a comprehensive blog post by synthesizing the 'positive_text', 'negative_text'"
            ", and 'neutral_text' from the context. The post should present a balanced view on the "
            "topic of Context Engineering."
        ),
        depends_on=["read_positive_texts", "read_negative_texts", "read_neutral_texts"],
        condition=ConditionDSL.combine_and(
            ConditionDSL.data_exists("read_positive_texts"),
            ConditionDSL.data_exists("read_negative_texts"),
            ConditionDSL.data_exists("read_neutral_texts"),
        ),
        output_schema=BlogPostOutput
    ).step(
        name="publish_post",
        goal="Publish the post by saving the content to a file.",
        depends_on=["post_creation"],
        tools_filter=["save_post"],
        condition=ConditionDSL.data_exists("post_creation.post")
    ).build()

    # Configure TAgent
    config = TAgentConfig(
        model="openrouter/google/gemini-2.5-flash-lite-preview-06-17",
        tools={
            "read_positive_text": read_positive_text,
            "read_negative_text": read_negative_text,
            "read_neutral_text": read_neutral_text,
            "save_post": save_post,
        },
        verbose=False,
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
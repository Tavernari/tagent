
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field
import random

from tagent import run_agent
from tagent.pipeline import PipelineBuilder, ConditionDSL, ExecutionMode

from os import path

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

pipeline = PipelineBuilder(
    "post_creator_pipeline",
    "A pipeline to create a post from texts using an LLM for content generation.",
)

pipeline.step(
    name="read_positive_texts",
    goal="Load the positive text",
    execution_mode=ExecutionMode.CONCURRENT,
    tools_filter=["read_positive_text"],
)

pipeline.step(
    name="read_negative_texts",
    goal="Load the negative text",
    execution_mode=ExecutionMode.CONCURRENT,
    tools_filter=["read_negative_text"],
)

pipeline.step(
    name="read_neutral_texts",
    goal="Load the neutral text",
    execution_mode=ExecutionMode.CONCURRENT,
    tools_filter=["read_neutral_text"],
)

# This step now uses the LLM directly for content creation
pipeline.step(
    name="post_creation",
    goal="Create a comprehensive blog post by synthesizing the 'positive_text', 'negative_text', and 'neutral_text' from the context. The post should present a balanced view on the topic of Context Engineering.",
    depends_on=["read_positive_texts", "read_negative_texts", "read_neutral_texts"],
    condition=ConditionDSL.combine_and(
        ConditionDSL.data_exists("read_positive_texts"),
        ConditionDSL.data_exists("read_negative_texts"),
        ConditionDSL.data_exists("read_neutral_texts"),
    )
    
)

pipeline.step(
    name="publish_post",
    goal="Publish the post by saving the content to a file.",
    depends_on=["post_creation"],
    tools_filter=["save_post"],
    condition=ConditionDSL.data_exists("post_creation")
)

# The 'create_post' tool is removed as it's no longer needed
agent_response = run_agent(
    goal_or_pipeline=pipeline.build(),
    model="openrouter/google/gemini-2.5-flash-lite-preview-06-17",
    tools={
        "read_positive_text": read_positive_text,
        "read_negative_text": read_negative_text,
        "read_neutral_text": read_neutral_text,
        "save_post": save_post,
    },
    verbose=True,
)

print(agent_response)

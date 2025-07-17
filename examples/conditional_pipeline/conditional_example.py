#!/usr/bin/env python3
"""
Example demonstrating the new expressive condition classes in TAgent pipelines.

This example shows how to use IsGreaterThan, IsLessThan, IsEqualTo, Contains, 
And, Or, Not conditions for sophisticated conditional execution.
"""

import asyncio
from typing import List, Tuple
from pydantic import BaseModel, Field

from tagent.pipeline import PipelineBuilder
from tagent.pipeline.conditions import (
    IsGreaterThan, IsLessThan, IsEmpty,
    And, Or, Not, Conditions
)
from tagent.pipeline.executor import PipelineExecutor, PipelineExecutorConfig
from tagent.config import TAgentConfig


# Define structured output schemas
class SentimentAnalysis(BaseModel):
    """Schema for sentiment analysis results."""
    score: float = Field(description="Sentiment score from 0 to 10")
    confidence: float = Field(description="Confidence level (0-1)")
    category: str = Field(description="Sentiment category: positive, negative, neutral")
    keywords: List[str] = Field(description="Key emotional words found")


class ResponseAction(BaseModel):
    """Schema for response actions."""
    action_type: str = Field(description="Type of action: email, escalation, standard")
    message: str = Field(description="Response message or action description")
    priority: str = Field(description="Priority level: low, medium, high, critical")


# Tool functions with proper TAgent signatures
def analyze_sentiment(feedback: str) -> Tuple[str, SentimentAnalysis]:
    """
    Analyze sentiment of customer feedback.
    
    Args:
        feedback (str): Customer feedback text.
    
    Returns:
        Tuple[str, SentimentAnalysis]: Sentiment analysis result.
    """
    positive_words = ["great", "excellent", "amazing", "love", "fantastic", "perfect"]
    negative_words = ["terrible", "awful", "hate", "worst", "bad", "disappointed"]
    
    words = feedback.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count > negative_count:
        score = 7.0 + min(positive_count * 0.5, 3.0)
        category = "positive"
    elif negative_count > positive_count:
        score = 3.0 - min(negative_count * 0.5, 2.5)
        category = "negative"
    else:
        score = 5.0
        category = "neutral"
    
    confidence = min(0.9, max(0.3, abs(positive_count - negative_count) * 0.2 + 0.3))
    
    keywords = [word for word in words if word in positive_words or word in negative_words]
    
    result = SentimentAnalysis(
        score=score,
        confidence=confidence,
        category=category,
        keywords=keywords
    )
    
    return ("sentiment_analysis", result)


def send_positive_response() -> Tuple[str, ResponseAction]:
    """Send positive response for good feedback."""
    return ("response_action", ResponseAction(
        action_type="email",
        message="Thank you for your positive feedback! We're delighted to hear about your experience.",
        priority="low"
    ))


def escalate_negative_feedback() -> Tuple[str, ResponseAction]:
    """Escalate negative feedback to management."""
    return ("response_action", ResponseAction(
        action_type="escalation",
        message="Negative feedback escalated to management for immediate attention.",
        priority="high"
    ))


def send_neutral_response() -> Tuple[str, ResponseAction]:
    """Send standard response for neutral feedback."""
    return ("response_action", ResponseAction(
        action_type="standard",
        message="Thank you for your feedback. We appreciate your input and will use it to improve our service.",
        priority="medium"
    ))


def follow_up_low_confidence() -> Tuple[str, ResponseAction]:
    """Follow up when confidence is low."""
    return ("response_action", ResponseAction(
        action_type="follow_up",
        message="Following up on feedback with low confidence analysis - manual review required.",
        priority="medium"
    ))


async def main():
    """Main function demonstrating conditional pipeline execution."""
    print("\n🎯 CONDITIONAL PIPELINE EXAMPLE")
    print("=" * 50)
    
    # Create pipeline with sophisticated conditional logic
    pipeline = PipelineBuilder(
        name="conditional_feedback_pipeline",
        description="Process customer feedback with conditional routing based on sentiment analysis"
    ).step(
        name="analyze_feedback",
        goal="Analyze the sentiment of customer feedback and categorize it.",
        tools=[analyze_sentiment],
        output_schema=SentimentAnalysis,
    ).step(
        name="positive_response",
        goal="Send positive response for satisfied customers.",
        depends_on=["analyze_feedback"],
        # Condition: score > 7 AND confidence > 0.6
        condition=And(
            IsGreaterThan("analyze_feedback.score", 7.0),
            IsGreaterThan("analyze_feedback.confidence", 0.6)
        ),
        tools=[send_positive_response]
    ).step(
        name="negative_escalation",
        goal="Escalate negative feedback to management.",
        depends_on=["analyze_feedback"],
        # Condition: score < 4 OR contains negative keywords
        condition=Or(
            IsLessThan("analyze_feedback.score", 4.0),
            Not(IsEmpty("analyze_feedback.keywords"))
        ),
        tools=[escalate_negative_feedback]
    ).step(
        name="neutral_response",
        goal="Send standard response for neutral feedback.",
        depends_on=["analyze_feedback"],
        # Condition: score between 4-7 AND not negative keywords
        condition=And(
            IsGreaterThan("analyze_feedback.score", 4.0),
            IsLessThan("analyze_feedback.score", 7.0),
            IsEmpty("analyze_feedback.keywords")  # No strong emotional keywords
        ),
        tools=[send_neutral_response]
    ).step(
        name="low_confidence_followup",
        goal="Follow up when analysis confidence is low.",
        depends_on=["analyze_feedback"],
        # Condition: confidence < 0.5 (regardless of score)
        condition=IsLessThan("analyze_feedback.confidence", 0.5),
        tools=[follow_up_low_confidence]
    ).step(
        name="high_confidence_positive",
        goal="Special handling for high-confidence positive feedback.",
        depends_on=["analyze_feedback"],
        # Condition: positive category AND high confidence AND good score
        condition=Conditions.all_of(
            Conditions.is_equal_to("analyze_feedback.category", "positive"),
            Conditions.is_greater_than("analyze_feedback.confidence", 0.8),
            Conditions.is_greater_than("analyze_feedback.score", 8.0)
        ),
        tools=[send_positive_response]
    ).build()
    
    # Configure TAgent
    config = TAgentConfig(
        model="openrouter/google/gemini-2.5-flash-lite-preview-06-17",
        verbose=True
    )
    
    # Configure executor
    executor_config = PipelineExecutorConfig(
        max_concurrent_steps=2,
        enable_persistence=False
    )
    
    # Test with different types of feedback
    test_cases = [
        "The service was absolutely fantastic! I love everything about it!",
        "This was terrible. I hate the experience completely.",
        "The service was okay, nothing special.",
        "Good enough I guess"  # Low confidence case
    ]
    
    for i, feedback in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {feedback}")
        print("-" * 60)
        
        # Add feedback to pipeline context
        pipeline.get_step(name="analyze_feedback").append_goal(feedback)
        
        # Execute pipeline
        executor = PipelineExecutor(pipeline, config, executor_config)
        result = await executor.execute()
        
        print(f"✅ Pipeline completed successfully: {result.success}")
        if result.success:
            print(f"📊 Steps completed: {result.steps_completed}")
            
            # Show which steps executed
            for step_name, step_result in result.step_outputs.items():
                if step_result and hasattr(step_result, 'result'):
                    print(f"  → {step_name}: {step_result.result}")
        else:
            print(f"❌ Pipeline failed: {result.error_details}")
        
        print()


if __name__ == "__main__":
    asyncio.run(main())
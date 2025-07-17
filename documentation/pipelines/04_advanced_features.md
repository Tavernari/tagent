# 4. Advanced Pipeline Features

Once you've mastered the core concepts, you can leverage TAgent's advanced features to build more efficient, robust, and dynamic pipelines.

## Step-Specific Tools

While you can provide a global list of tools in the `TAgentConfig`, you can also provide a specific list of tools for an individual step. This is useful for isolating functionality and improving security by ensuring a step only has access to the tools it needs.

Step-specific tools are **combined** with global tools. If a tool with the same name exists in both lists, the step-specific version will be used.

```python
# Define a global tool
def global_tool():
    """A tool available to all steps."""
    return "global_result", "from global"

# Define a step-specific tool
def specialized_tool():
    """A tool only for a specific step."""
    return "special_result", "from specialized"

# In the PipelineBuilder...
builder.step(
    name="special_step",
    goal="This step needs a special tool.",
    tools=[specialized_tool]  # Provide the tool directly to the step
)

# In the main run call...
agent_config = TAgentConfig(
    model="gpt-4o-mini",
    tools=[global_tool] # Global tools
)

executor = PipelineExecutor(pipeline, config=agent_config)
```
In this scenario, the `special_step` will have access to both `specialized_tool` and `global_tool`. Other steps in the pipeline will only have access to `global_tool`.

## Concurrent Execution

By default, steps run in `SERIAL` mode. However, you can run independent steps in `CONCURRENT` mode to save time.

In this example, we'll research a topic and then *simultaneously* write a blog post and a tweet about it.

```python
# In the PipelineBuilder...
builder.step(
    name="research",
    goal="Research the benefits of a 4-day work week."
).step(
    name="write_blog_post",
    goal="Write a 300-word blog post based on the research.",
    depends_on=["research"],
    execution_mode=ExecutionMode.CONCURRENT # This can run in parallel
).step(
    name="write_tweet",
    goal="Write a catchy tweet based on the research.",
    depends_on=["research"],
    execution_mode=ExecutionMode.CONCURRENT # This can also run in parallel
).step(
    name="final_summary",
    goal="Combine the blog post and the tweet into a final summary document.",
    # This step depends on BOTH parallel steps, creating a "fan-in"
    depends_on=["write_blog_post", "write_tweet"]
)
```

The execution graph looks like this:

```
           /--> write_blog_post --\
research --<                        >--> final_summary
           \--> write_tweet   ----/
```

The `PipelineExecutor` will automatically run `write_blog_post` and `write_tweet` at the same time after `research` is complete, potentially cutting down the total execution time.

*Note: You must import `ExecutionMode` from `tagent.pipeline.models`.*

## Data Flow Control with `read_data`

One of the most powerful features of TAgent pipelines is the ability to explicitly read and inject outputs from previous steps into subsequent steps. This is achieved through the `read_data` parameter, which allows you to:

- **Enhance prompts** with specific outputs from previous steps
- **Automatically inject parameters** into tool functions
- **Create sophisticated data pipelines** with precise control over information flow

### Basic Usage

The `read_data` parameter accepts a list of strings that specify which outputs to read from previous steps:

```python
from pydantic import BaseModel, Field

# Define the expected output structure for the analysis step
class AnalysisOutput(BaseModel):
    insights: str = Field(description="Key insights extracted from the data")
    summary: str = Field(description="Summary of the analysis results")
    confidence: float = Field(description="Confidence score of the analysis")

builder.step(
    name="analyze_data",
    goal="Analyze raw data and extract insights.",
    tools=[data_analyzer],
    output_schema=AnalysisOutput  # Define the structured output
).step(
    name="create_report",
    goal="Create a comprehensive report based on the analysis.",
    depends_on=["analyze_data"],
    read_data=["analyze_data.insights", "analyze_data.summary"],  # Now these fields are defined
    tools=[report_generator]
)
```

### Automatic Parameter Injection

When you use `read_data`, TAgent automatically injects the specified values as parameters to your tool functions. This enables seamless data flow without manual parameter passing:

```python
from typing import Tuple
from pydantic import BaseModel, Field

# Define the output schema for content generation
class BlogPostOutput(BaseModel):
    post: str = Field(description="The generated blog post content")
    title: str = Field(description="The blog post title")
    word_count: int = Field(description="Number of words in the post")

def save_post(state: dict, args: dict) -> Tuple[str, str]:
    """
    Saves the generated post to a file.
    
    The 'post' parameter will be automatically injected from read_data
    when this tool is called by a step with read_data=["generate_content.post"]
    """
    post_content = args.get("post", "")
    
    if not post_content:
        return ("error", "No post content provided")
    
    with open("output.md", "w") as f:
        f.write(post_content)
    return ("result", "Post saved successfully")

# In your pipeline
builder.step(
    name="generate_content",
    goal="Generate a comprehensive blog post about AI trends.",
    output_schema=BlogPostOutput  # Defines the structured output with 'post' field
).step(
    name="save_content",
    goal="Save the generated content to a file.",
    depends_on=["generate_content"],
    read_data=["generate_content.post"],  # This will be injected as 'post' parameter
    tools=[save_post]
)
```

### Real-World Example: Blog Post Creation

Here's a complete example showing how `read_data` enables sophisticated content creation workflows:

```python
from typing import Tuple
from tagent.pipeline import PipelineBuilder
from tagent.pipeline.models import ExecutionMode
from pydantic import BaseModel, Field
import os

# Define strongly typed output schemas
class TextExtracted(BaseModel):
    """Schema for text extraction steps."""
    text: str = Field(description="The extracted text content from the source")
    source_file: str = Field(description="Path to the source file")
    word_count: int = Field(description="Number of words in the extracted text")

class BlogPostOutput(BaseModel):
    """Schema for blog post generation step."""
    post: str = Field(description="The complete generated blog post content")
    title: str = Field(description="The blog post title")
    word_count: int = Field(description="Total word count of the post")
    sections: list = Field(description="List of section titles in the post")

# Define typed tool functions
def read_positive_text(state: dict, args: dict) -> Tuple[str, TextExtracted]:
    """
    Reads positive perspective text from file.
    Returns structured output that matches TextExtracted schema.
    """
    file_path = "texts/positive.md"
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        result = TextExtracted(
            text=content,
            source_file=file_path,
            word_count=len(content.split())
        )
        return ("positive_text", result)
    except Exception as e:
        # Return error but maintain schema structure
        return ("positive_text", TextExtracted(
            text="",
            source_file=file_path,
            word_count=0
        ))

def read_negative_text(state: dict, args: dict) -> Tuple[str, TextExtracted]:
    """Reads negative perspective text from file."""
    file_path = "texts/negative.md"
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        result = TextExtracted(
            text=content,
            source_file=file_path,
            word_count=len(content.split())
        )
        return ("negative_text", result)
    except Exception as e:
        return ("negative_text", TextExtracted(
            text="",
            source_file=file_path,
            word_count=0
        ))

def read_neutral_text(state: dict, args: dict) -> Tuple[str, TextExtracted]:
    """Reads neutral perspective text from file."""
    file_path = "texts/neutral.md"
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        result = TextExtracted(
            text=content,
            source_file=file_path,
            word_count=len(content.split())
        )
        return ("neutral_text", result)
    except Exception as e:
        return ("neutral_text", TextExtracted(
            text="",
            source_file=file_path,
            word_count=0
        ))

def save_post(state: dict, args: dict) -> Tuple[str, str]:
    """
    Saves the generated blog post to a file.
    
    The 'post' parameter is automatically injected from read_data=["synthesize_content.post"]
    This demonstrates how read_data enables seamless parameter injection.
    """
    post_content = args.get("post", "")
    
    if not post_content:
        return ("save_result", "Failed: No post content provided")
    
    output_path = "output/final_post.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, "w") as f:
            f.write(post_content)
        return ("save_result", f"Post saved successfully to {output_path}")
    except Exception as e:
        return ("save_result", f"Failed to save post: {str(e)}")

# Create a pipeline that reads multiple text sources and synthesizes them
pipeline = PipelineBuilder(
    name="blog_creation_pipeline",
    description="Creates a balanced blog post from multiple text sources with type safety."
).step(
    name="load_positive_content",
    goal="Load positive perspective content from file and extract metadata.",
    tools=[read_positive_text],
    output_schema=TextExtracted,  # Ensures type-safe output
    execution_mode=ExecutionMode.CONCURRENT
).step(
    name="load_negative_content", 
    goal="Load negative perspective content from file and extract metadata.",
    tools=[read_negative_text],
    output_schema=TextExtracted,  # Ensures type-safe output
    execution_mode=ExecutionMode.CONCURRENT
).step(
    name="load_neutral_content",
    goal="Load neutral perspective content from file and extract metadata.", 
    tools=[read_neutral_text],
    output_schema=TextExtracted,  # Ensures type-safe output
    execution_mode=ExecutionMode.CONCURRENT
).step(
    name="synthesize_content",
    goal="Create a balanced blog post synthesizing all perspectives into a cohesive narrative.",
    depends_on=["load_positive_content", "load_negative_content", "load_neutral_content"],
    # Read specific typed outputs from previous steps to enhance the LLM prompt
    read_data=[
        "load_positive_content.text",    # String: positive text content
        "load_negative_content.text",    # String: negative text content
        "load_neutral_content.text"      # String: neutral text content
    ],
    output_schema=BlogPostOutput  # Ensures structured, typed output
).step(
    name="publish_post",
    goal="Save the final blog post to a file in the output directory.",
    depends_on=["synthesize_content"],
    read_data=["synthesize_content.post"],  # Auto-inject 'post' field as parameter
    tools=[save_post]
).build()
```

**Key Benefits of This Approach:**

1. **Type Safety**: Each step has a defined `output_schema` ensuring consistent, typed outputs
2. **Parameter Injection**: The `read_data` automatically injects values into tool functions
3. **Prompt Enhancement**: LLM steps receive rich context from previous structured outputs
4. **Concurrent Execution**: Text loading steps run in parallel for better performance
5. **Error Handling**: Tools handle failures gracefully while maintaining schema compliance

### Advanced Data Path Specifications

The `read_data` parameter supports flexible path specifications with full type safety:

```python
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Define comprehensive output schemas
class UserAnalysis(BaseModel):
    """Schema for user analysis step."""
    sentiment_score: float = Field(description="Sentiment score from -1 to 1")
    key_themes: List[str] = Field(description="List of identified themes")
    user_segments: Dict[str, int] = Field(description="User segments with counts")
    raw_feedback: str = Field(description="Original feedback text")

class MarketData(BaseModel):
    """Schema for market research step."""
    market_size: float = Field(description="Total addressable market size")
    competitor_analysis: Dict[str, Any] = Field(description="Competitor data")
    trends: List[str] = Field(description="Market trends")

class RecommendationOutput(BaseModel):
    """Schema for final recommendation step."""
    recommendations: List[str] = Field(description="List of actionable recommendations")
    priority_matrix: Dict[str, str] = Field(description="Priority levels for each recommendation")
    implementation_plan: str = Field(description="Detailed implementation plan")

# Examples of different read_data patterns:

# 1. Read entire step output (gets the complete schema object)
builder.step(
    name="process_full_analysis",
    goal="Process the complete user analysis results.",
    depends_on=["user_analysis"],
    read_data=["user_analysis"],  # Gets complete UserAnalysis object
    tools=[process_analysis]
)

# 2. Read specific typed attributes from structured outputs
builder.step(
    name="create_sentiment_report",
    goal="Create a report focusing on sentiment and themes.",
    depends_on=["user_analysis"],
    read_data=[
        "user_analysis.sentiment_score",  # float
        "user_analysis.key_themes",       # List[str]
        "user_analysis.user_segments"     # Dict[str, int]
    ],
    tools=[create_report]
)

# 3. Read from multiple steps with different data types
builder.step(
    name="generate_recommendations",
    goal="Generate business recommendations based on user and market data.",
    depends_on=["user_analysis", "market_research"],
    read_data=[
        "user_analysis.sentiment_score",      # float
        "user_analysis.key_themes",           # List[str]
        "market_research.market_size",        # float
        "market_research.competitor_analysis", # Dict[str, Any]
        "market_research.trends"              # List[str]
    ],
    output_schema=RecommendationOutput,
    tools=[recommendation_generator]
)

# 4. Complex nested attribute access
class DetailedAnalysis(BaseModel):
    summary: str = Field(description="Analysis summary")
    details: Dict[str, Any] = Field(description="Detailed analysis data")
    metadata: Dict[str, str] = Field(description="Analysis metadata")

builder.step(
    name="extract_specific_data",
    goal="Extract specific nested data from detailed analysis.",
    depends_on=["detailed_analysis"],
    read_data=[
        "detailed_analysis.summary",           # str
        "detailed_analysis.details",          # Dict[str, Any] - entire dict
        "detailed_analysis.metadata"          # Dict[str, str] - entire dict
    ],
    tools=[data_extractor]
)
```

**Data Path Resolution Rules:**

1. **`"step_name"`** - Returns the complete output object from the step
2. **`"step_name.attribute"`** - Returns the specific attribute value with its original type
3. **Type preservation** - All data types from Pydantic schemas are preserved during injection
4. **Nested access** - Complex nested structures are fully supported
5. **Multiple sources** - Can combine data from multiple steps in a single read_data list

### Benefits of `read_data`

1. **Precise Control**: Explicitly define which data flows between steps
2. **Enhanced Prompts**: LLM steps receive richer context from previous outputs
3. **Automatic Injection**: Tool functions receive parameters without manual wiring
4. **Type Safety**: Works seamlessly with Pydantic schemas and structured outputs
5. **Debugging**: Clear visibility into data dependencies and flow

### Best Practices for Typed Tool Functions

When implementing tool functions with `read_data`, follow these best practices for maximum type safety and maintainability:

```python
from typing import Tuple, Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import logging
import time

# Define comprehensive schemas with validation
class DataAnalysisInput(BaseModel):
    """Input schema for data analysis tools."""
    raw_data: str = Field(description="Raw data to analyze")
    analysis_type: str = Field(description="Type of analysis to perform")
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed_types = ['sentiment', 'classification', 'summarization']
        if v not in allowed_types:
            raise ValueError(f'Analysis type must be one of: {allowed_types}')
        return v

class DataAnalysisOutput(BaseModel):
    """Output schema for data analysis results."""
    analysis_result: str = Field(description="Primary analysis result")
    confidence_score: float = Field(description="Confidence score (0-1)", ge=0, le=1)
    metadata: Dict[str, Any] = Field(description="Additional analysis metadata")
    processing_time: float = Field(description="Processing time in seconds")

def analyze_data_with_validation(state: dict, args: dict) -> Tuple[str, DataAnalysisOutput]:
    """
    Performs data analysis with comprehensive input validation and typed outputs.
    
    This function demonstrates best practices for tool implementation:
    - Input validation using Pydantic models
    - Proper error handling with fallback values
    - Structured, typed output
    - Logging for debugging
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input parameters (can be injected via read_data)
        input_data = DataAnalysisInput(
            raw_data=args.get('raw_data', ''),
            analysis_type=args.get('analysis_type', 'sentiment')
        )
        
        logger.info(f"Starting {input_data.analysis_type} analysis")
        
        # Perform the actual analysis
        start_time = time.time()
        
        if input_data.analysis_type == 'sentiment':
            result = perform_sentiment_analysis(input_data.raw_data)
        elif input_data.analysis_type == 'classification':
            result = perform_classification(input_data.raw_data)
        else:
            result = perform_summarization(input_data.raw_data)
        
        processing_time = time.time() - start_time
        
        # Return structured, validated output
        output = DataAnalysisOutput(
            analysis_result=result['text'],
            confidence_score=result['confidence'],
            metadata={'model_version': '1.0', 'technique': input_data.analysis_type},
            processing_time=processing_time
        )
        
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        return ("analysis_result", output)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        
        # Return error state with valid schema
        error_output = DataAnalysisOutput(
            analysis_result="Analysis failed",
            confidence_score=0.0,
            metadata={'error': str(e)},
            processing_time=0.0
        )
        return ("analysis_result", error_output)

# Example of using this tool in a pipeline with read_data
class TextProcessingOutput(BaseModel):
    """Schema for text processing step."""
    processed_text: str = Field(description="Processed text content")
    original_length: int = Field(description="Original text length")
    processed_length: int = Field(description="Processed text length")

pipeline = PipelineBuilder(
    name="advanced_text_pipeline",
    description="Advanced text processing with full type safety"
).step(
    name="preprocess_text",
    goal="Clean and preprocess raw text data.",
    tools=[text_preprocessor],
    output_schema=TextProcessingOutput
).step(
    name="analyze_processed_text",
    goal="Perform comprehensive analysis on the processed text.",
    depends_on=["preprocess_text"],
    read_data=[
        "preprocess_text.processed_text",  # str - will be injected as 'raw_data'
        # Note: The parameter name mapping happens automatically
        # 'processed_text' -> 'raw_data' based on function signature
    ],
    tools=[analyze_data_with_validation],
    output_schema=DataAnalysisOutput
).step(
    name="generate_report",
    goal="Generate a comprehensive report from analysis results.",
    depends_on=["analyze_processed_text"],
    read_data=[
        "analyze_processed_text.analysis_result",    # str
        "analyze_processed_text.confidence_score",   # float
        "analyze_processed_text.metadata"           # Dict[str, Any]
    ],
    tools=[report_generator]
).build()
```

**Key Best Practices Demonstrated:**

1. **Input Validation**: Use Pydantic models to validate tool inputs
2. **Comprehensive Error Handling**: Always return valid schema objects, even on errors
3. **Logging**: Include proper logging for debugging and monitoring
4. **Type Annotations**: Use detailed type hints for all function signatures
5. **Schema Documentation**: Provide clear descriptions for all fields
6. **Validation Rules**: Use Pydantic validators for business logic validation
7. **Fallback Values**: Provide sensible defaults for all parameters

## Timeouts and Retries

You can make your pipeline more robust by setting timeouts and retry policies for steps that might be slow or fail intermittently.

```python
# In the .step() definition...
builder.step(
    name="fetch_external_api",
    goal="Fetch data from a potentially unreliable external API.",
    timeout=120,       # Abort this step if it takes longer than 120 seconds
    max_retries=3      # If the step fails, retry it up to 3 times
)
```

The `PipelineExecutor` will automatically manage the retry logic. If the step fails more than `max_retries`, the entire pipeline will be marked as failed.

## Conditional Execution

The TAgent pipeline system now supports sophisticated conditional execution using expressive condition classes. You can execute or skip steps based on the output of previous steps using intuitive, type-safe conditions.

### Basic Conditional Logic

```python
from tagent.pipeline.conditions import IsGreaterThan, IsLessThan, IsEqualTo, DataExists, And, Or, Not
from pydantic import BaseModel, Field

# Define structured output for sentiment analysis
class SentimentOutput(BaseModel):
    score: float = Field(description="Sentiment score from 0 to 10")
    confidence: float = Field(description="Confidence level (0-1)")
    category: str = Field(description="Sentiment category: positive, negative, neutral")

# Build a pipeline with conditional branching
builder.step(
    name="analyze_sentiment",
    goal="Analyze the sentiment of a user review and return a score from 0 to 10.",
    output_schema=SentimentOutput
).step(
    name="send_thank_you_email",
    goal="Draft a thank you email for the positive review.",
    depends_on=["analyze_sentiment"],
    # This step only runs if sentiment score is greater than 7
    condition=IsGreaterThan(path="analyze_sentiment.score", value=7.0)
).step(
    name="escalate_to_support",
    goal="Draft a support ticket to address the negative review.",
    depends_on=["analyze_sentiment"],
    # This step only runs if sentiment score is less than 4
    condition=IsLessThan(path="analyze_sentiment.score", value=4.0)
).step(
    name="send_neutral_response",
    goal="Send a standard response for neutral feedback.",
    depends_on=["analyze_sentiment"],
    # This step runs for neutral scores (4-7 range)
    condition=And(
        IsGreaterThan(path="analyze_sentiment.score", value=4.0),
        IsLessThan(path="analyze_sentiment.score", value=7.0)
    )
)
```

### Advanced Conditional Examples

```python
from tagent.pipeline.conditions import Contains, IsEmpty, IsEqualTo, IsGreaterThan, IsLessThan, And, Or, Not
from typing import Dict, List, Any

# Define complex output schema
class ProcessingResult(BaseModel):
    status: str = Field(description="Processing status: success, warning, error")
    errors: List[str] = Field(description="List of error messages")
    data: Dict[str, Any] = Field(description="Processed data")
    metrics: Dict[str, float] = Field(description="Processing metrics")

# Using the condition classes directly
builder.step(
    name="process_data",
    goal="Process uploaded data and return results.",
    output_schema=ProcessingResult
).step(
    name="handle_errors",
    goal="Handle any errors that occurred during processing.",
    depends_on=["process_data"],
    # Run only if there are errors
    condition=Not(IsEmpty("process_data.errors"))
).step(
    name="send_success_notification",
    goal="Send success notification to user.",
    depends_on=["process_data"],
    # Run only if status is success and no errors
    condition=And(
        IsEqualTo("process_data.status", "success"),
        IsEmpty("process_data.errors")
    )
).step(
    name="generate_detailed_report",
    goal="Generate a detailed processing report.",
    depends_on=["process_data"],
    # Run only if confidence metric is high
    condition=IsGreaterThan("process_data.metrics.confidence", 0.8)
).step(
    name="quality_assurance_check",
    goal="Perform additional quality assurance checks.",
    depends_on=["process_data"],
    # Run if status contains "warning" OR confidence is low
    condition=Or(
        Contains("process_data.status", "warning"),
        IsLessThan("process_data.metrics.confidence", 0.6)
    )
)
```

### Available Condition Classes

The pipeline system provides several expressive condition classes:

#### Basic Conditions
- **`DataExists(path)`** - Check if data exists at a path
- **`IsGreaterThan(path, value)`** - Numeric greater than comparison
- **`IsLessThan(path, value)`** - Numeric less than comparison
- **`IsEqualTo(path, value)`** - Equality comparison
- **`Contains(path, value)`** - Check if container contains value
- **`IsEmpty(path)`** - Check if value is empty (None, empty string, empty list)

#### Logical Operators
- **`And(*conditions)`** - All conditions must be true
- **`Or(*conditions)`** - Any condition must be true
- **`Not(condition)`** - Negates the condition

#### Usage Notes
- All conditions support dot notation for accessing nested data (e.g., `"step_name.attribute"`)
- Conditions are evaluated against the pipeline execution context
- Type-safe evaluation with graceful handling of missing or invalid data
- Can be combined with logical operators for complex conditional logic

### Real-World Example: Customer Feedback Pipeline

```python
from typing import List
from tagent.pipeline.conditions import IsGreaterThan, IsEqualTo, Contains, And, Or, Not

class FeedbackAnalysis(BaseModel):
    sentiment_score: float = Field(description="Sentiment score 0-10")
    urgency_level: str = Field(description="low, medium, high, critical")
    keywords: List[str] = Field(description="Extracted keywords")
    customer_tier: str = Field(description="bronze, silver, gold, platinum")

# Build conditional feedback processing pipeline
pipeline = PipelineBuilder(
    name="customer_feedback_pipeline",
    description="Process customer feedback with conditional routing"
).step(
    name="analyze_feedback",
    goal="Analyze customer feedback for sentiment, urgency, and keywords.",
    tools=[feedback_analyzer],
    output_schema=FeedbackAnalysis
).step(
    name="escalate_to_manager",
    goal="Escalate critical issues to management immediately.",
    depends_on=["analyze_feedback"],
    # Escalate if critical urgency OR very negative sentiment from premium customers
    condition=Or(
        IsEqualTo("analyze_feedback.urgency_level", "critical"),
        And(
            IsLessThan("analyze_feedback.sentiment_score", 3.0),
            Contains("analyze_feedback.customer_tier", "gold")
        )
    ),
    tools=[escalation_manager]
).step(
    name="auto_respond_positive",
    goal="Send automated positive response for high satisfaction.",
    depends_on=["analyze_feedback"],
    # Auto-respond to positive feedback from any customer
    condition=And(
        IsGreaterThan("analyze_feedback.sentiment_score", 8.0),
        IsEqualTo("analyze_feedback.urgency_level", "low")
    ),
    tools=[auto_responder]
).step(
    name="assign_to_specialist",
    goal="Assign technical issues to specialist team.",
    depends_on=["analyze_feedback"],
    # Assign to specialist if contains technical keywords
    condition=Or(
        Contains("analyze_feedback.keywords", "bug"),
        Contains("analyze_feedback.keywords", "error"),
        Contains("analyze_feedback.keywords", "integration")
    ),
    tools=[specialist_router]
).step(
    name="standard_response",
    goal="Send standard response for routine feedback.",
    depends_on=["analyze_feedback"],
    # Default case: not critical, not super positive, not technical
    condition=And(
        Not(IsEqualTo("analyze_feedback.urgency_level", "critical")),
        IsLessThan("analyze_feedback.sentiment_score", 8.0),
        IsGreaterThan("analyze_feedback.sentiment_score", 3.0),
        Not(Contains("analyze_feedback.keywords", "bug"))
    ),
    tools=[standard_responder]
).build()
```

This conditional execution system allows you to build sophisticated, branching workflows that adapt to the results of previous steps, making your pipelines more intelligent and responsive to different scenarios.

---

Finally, let's look at the key classes that make all of this possible.

➡️ [Next: API Reference](./05_api_reference.md)

```
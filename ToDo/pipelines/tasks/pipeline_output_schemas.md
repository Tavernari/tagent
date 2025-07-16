# Pipeline Output Schemas & Structured Results

## Overview
Each pipeline step can define its own output schema using Pydantic models, allowing for type-safe communication between steps and structured final results. This approach enables better data validation, clearer interfaces, and more robust pipeline execution.

## Core Concepts

### 1. Per-Step Output Schemas
Each pipeline step can define its own output schema, enabling structured and validated results:

```python
from pydantic import BaseModel, Field

class SearchResultOutput(BaseModel):
    """Output schema for web search step."""
    query: str = Field(..., description="The search query used")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    sources: List[str] = Field(..., description="List of sources used")
    confidence: float = Field(..., description="Confidence score of results")
    
class AnalysisOutput(BaseModel):
    """Output schema for analysis step."""
    summary: str = Field(..., description="Analysis summary")
    key_findings: List[str] = Field(..., description="Key findings")
    sentiment_score: float = Field(..., description="Sentiment analysis score")
    categories: List[str] = Field(..., description="Content categories")
    requires_deep_analysis: bool = Field(..., description="Whether deep analysis is needed")

class ReportOutput(BaseModel):
    """Output schema for final report step."""
    title: str = Field(..., description="Report title")
    executive_summary: str = Field(..., description="Executive summary")
    sections: List[Dict[str, Any]] = Field(..., description="Report sections")
    recommendations: List[str] = Field(..., description="Recommendations")
    appendices: Dict[str, Any] = Field(default_factory=dict, description="Additional data")
```

### 2. Enhanced Pipeline Step Definition
```python
@dataclass
class PipelineStep:
    """Enhanced pipeline step with output schema support."""
    name: str
    goal: str
    constraints: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SERIAL
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    tools_filter: Optional[List[str]] = None
    output_schema: Optional[Type[BaseModel]] = None  # NEW: Output schema definition
    
    def validate_output(self, result: Any) -> BaseModel:
        """Validate step output against schema."""
        if self.output_schema:
            if isinstance(result, dict):
                return self.output_schema(**result)
            elif isinstance(result, self.output_schema):
                return result
            else:
                raise ValueError(f"Step '{self.name}' output does not match schema {self.output_schema}")
        return result
```

### 3. Pipeline Result Structure
```python
@dataclass
class PipelineResult:
    """Enhanced pipeline result with structured outputs."""
    pipeline_name: str
    success: bool
    execution_time: float
    
    # Cost tracking (existing functionality)
    total_cost: float
    cost_per_step: Dict[str, float]
    token_usage: Dict[str, TokenUsage]
    
    # Step results with structured outputs
    step_outputs: Dict[str, BaseModel]  # Step name -> Structured output
    step_metadata: Dict[str, Dict[str, Any]]  # Step name -> Metadata
    
    # Final aggregated result (optional)
    final_output: Optional[BaseModel] = None
    
    # Memory and learning artifacts
    learned_facts: Dict[str, Any] = field(default_factory=dict)
    saved_memories: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    failed_steps: List[str] = field(default_factory=list)
    error_details: Dict[str, str] = field(default_factory=dict)
```

## Implementation Examples

### 1. Company Research Pipeline with Schemas
```python
# Define output schemas for each step
class CompanyBasicInfo(BaseModel):
    name: str
    industry: str
    location: str
    founded_year: Optional[int]
    employee_count: Optional[int]
    website: str
    description: str
    key_products: List[str]

class ReputationAnalysis(BaseModel):
    overall_score: float
    total_complaints: int
    resolved_complaints: int
    common_issues: List[str]
    satisfaction_rating: float
    recommendation_rate: float

class SocialMediaMetrics(BaseModel):
    platform: str
    followers: int
    engagement_rate: float
    recent_posts: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, float]
    brand_mentions: int

class CompanyReport(BaseModel):
    company_name: str
    research_date: str
    executive_summary: str
    basic_info: CompanyBasicInfo
    reputation: ReputationAnalysis
    social_media: Dict[str, SocialMediaMetrics]
    overall_rating: float
    recommendations: List[str]

# Create pipeline with output schemas
company_pipeline = Pipeline("company_research", "Comprehensive company analysis")

company_pipeline.step(
    name="basic_info",
    goal="Gather basic company information",
    output_schema=CompanyBasicInfo,
    constraints=["Use only reliable sources"]
).step(
    name="reputation_check",
    goal="Analyze company reputation on Reclame Aqui",
    depends_on=["basic_info"],
    output_schema=ReputationAnalysis,
    constraints=["Focus on recent complaints", "Calculate satisfaction metrics"]
).step(
    name="social_analysis",
    goal="Analyze social media presence",
    depends_on=["basic_info"],
    output_schema=SocialMediaMetrics,
    execution_mode=ExecutionMode.CONCURRENT
).step(
    name="final_report",
    goal="Generate comprehensive company report",
    depends_on=["basic_info", "reputation_check", "social_analysis"],
    output_schema=CompanyReport
)
```

### 2. Inter-Step Communication with Schemas
```python
class StepExecutionContext:
    """Enhanced context with schema-aware step communication."""
    
    def __init__(self, pipeline_memory: PipelineMemory):
        self.pipeline_memory = pipeline_memory
        self.typed_outputs: Dict[str, BaseModel] = {}
    
    def get_step_output(self, step_name: str, expected_type: Type[BaseModel]) -> BaseModel:
        """Get typed output from a previous step."""
        if step_name not in self.typed_outputs:
            raw_output = self.pipeline_memory.get_step_result(step_name)
            if raw_output and hasattr(raw_output, 'data'):
                # Validate and convert to expected type
                self.typed_outputs[step_name] = expected_type(**raw_output.data)
        
        return self.typed_outputs.get(step_name)
    
    def save_structured_output(self, step_name: str, output: BaseModel):
        """Save structured output with validation."""
        self.typed_outputs[step_name] = output
        self.pipeline_memory.save_step_result(step_name, {
            'data': output.model_dump(),
            'schema': output.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        })

# Usage in custom tools
def reputation_analysis_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """Tool that uses structured input and produces structured output."""
    context = StepExecutionContext(state.get('pipeline_memory'))
    
    # Get structured input from previous step
    company_info = context.get_step_output('basic_info', CompanyBasicInfo)
    
    # Perform analysis using structured data
    analysis_result = analyze_company_reputation(
        company_name=company_info.name,
        industry=company_info.industry,
        website=company_info.website
    )
    
    # Create structured output
    reputation_result = ReputationAnalysis(
        overall_score=analysis_result['score'],
        total_complaints=analysis_result['complaints'],
        resolved_complaints=analysis_result['resolved'],
        common_issues=analysis_result['issues'],
        satisfaction_rating=analysis_result['satisfaction'],
        recommendation_rate=analysis_result['recommendation_rate']
    )
    
    # Save structured output
    context.save_structured_output('reputation_check', reputation_result)
    
    return ('reputation_analysis', reputation_result.model_dump())
```

### 3. Memory and Learning Integration
```python
class PipelineLearningSystem:
    """System for capturing and using learning from pipeline execution."""
    
    def __init__(self):
        self.learned_facts: Dict[str, Any] = {}
        self.pattern_recognition: Dict[str, List[Any]] = {}
    
    def extract_learnings(self, step_output: BaseModel, step_name: str):
        """Extract learnings from structured step output."""
        if isinstance(step_output, CompanyBasicInfo):
            self._learn_company_patterns(step_output)
        elif isinstance(step_output, ReputationAnalysis):
            self._learn_reputation_patterns(step_output)
        elif isinstance(step_output, SocialMediaMetrics):
            self._learn_social_patterns(step_output)
    
    def _learn_company_patterns(self, company_info: CompanyBasicInfo):
        """Learn patterns from company information."""
        industry = company_info.industry
        
        if industry not in self.pattern_recognition:
            self.pattern_recognition[industry] = []
        
        self.pattern_recognition[industry].append({
            'employee_count': company_info.employee_count,
            'founded_year': company_info.founded_year,
            'key_products': company_info.key_products
        })
        
        # Store learned facts
        self.learned_facts[f"industry_{industry}_average_size"] = self._calculate_average_size(industry)
    
    def provide_context_for_step(self, step_name: str, current_outputs: Dict[str, BaseModel]) -> Dict[str, Any]:
        """Provide learned context for current step execution."""
        context = {}
        
        if step_name == "reputation_check" and "basic_info" in current_outputs:
            company_info = current_outputs["basic_info"]
            industry_patterns = self.pattern_recognition.get(company_info.industry, [])
            context["industry_reputation_patterns"] = industry_patterns
        
        return context
```

### 4. Pipeline Execution with Output Schemas
```python
class SchemaAwarePipelineExecutor(PipelineExecutor):
    """Pipeline executor with output schema support."""
    
    def __init__(self, pipeline: Pipeline, config: Dict[str, Any]):
        super().__init__(pipeline, config)
        self.learning_system = PipelineLearningSystem()
        self.output_validator = OutputValidator()
    
    async def _execute_single_step(self, step: PipelineStep):
        """Execute step with output schema validation."""
        async with self.executor_pool:
            try:
                # Prepare context with structured inputs
                context = await self._prepare_structured_context(step)
                
                # Execute step
                raw_result = await self._execute_tagent_step(step, context)
                
                # Validate and structure output
                if step.output_schema:
                    structured_result = step.validate_output(raw_result)
                    
                    # Extract learnings
                    self.learning_system.extract_learnings(structured_result, step.name)
                    
                    # Save structured result
                    self.state_machine.save_structured_step_result(step.name, structured_result)
                else:
                    # Save raw result for steps without schema
                    self.state_machine.save_step_result(step.name, raw_result)
                
                # Broadcast completion with structured data
                await self._broadcast_step_completion(step, structured_result)
                
            except Exception as e:
                await self._handle_step_error(step, e)
    
    async def _prepare_structured_context(self, step: PipelineStep) -> Dict[str, Any]:
        """Prepare execution context with structured inputs from dependencies."""
        context = {}
        
        # Add structured outputs from dependencies
        for dep_name in step.depends_on:
            dep_output = self.state_machine.get_structured_step_result(dep_name)
            if dep_output:
                context[f"input_{dep_name}"] = dep_output
        
        # Add learning context
        learning_context = self.learning_system.provide_context_for_step(
            step.name, 
            self.state_machine.get_all_structured_outputs()
        )
        context.update(learning_context)
        
        return context
    
    async def _create_pipeline_result(self) -> PipelineResult:
        """Create final pipeline result with structured outputs."""
        return PipelineResult(
            pipeline_name=self.pipeline.name,
            success=self.state_machine.is_successful(),
            execution_time=time.time() - self.start_time,
            total_cost=self._calculate_total_cost(),
            cost_per_step=self._calculate_cost_per_step(),
            token_usage=self._get_token_usage(),
            step_outputs=self.state_machine.get_all_structured_outputs(),
            step_metadata=self.state_machine.get_step_metadata(),
            final_output=self._create_final_output(),
            learned_facts=self.learning_system.learned_facts,
            saved_memories=self.state_machine.get_saved_memories(),
            failed_steps=self.state_machine.get_failed_steps(),
            error_details=self.state_machine.get_error_details()
        )
```

## Usage Examples

### 1. Accessing Structured Results
```python
# Execute pipeline
result = await run_pipeline(company_pipeline)

# Access structured outputs
basic_info = result.step_outputs['basic_info']  # CompanyBasicInfo object
reputation = result.step_outputs['reputation_check']  # ReputationAnalysis object
social_metrics = result.step_outputs['social_analysis']  # SocialMediaMetrics object

# Use structured data
print(f"Company: {basic_info.name}")
print(f"Industry: {basic_info.industry}")
print(f"Reputation Score: {reputation.overall_score}")
print(f"Social Media Followers: {social_metrics.followers}")

# Access final report if available
if result.final_output:
    final_report = result.final_output  # CompanyReport object
    print(f"Overall Rating: {final_report.overall_rating}")
    print(f"Recommendations: {final_report.recommendations}")
```

### 2. Creating Custom Report Step
```python
def create_final_report_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """Tool that aggregates all structured outputs into a final report."""
    context = StepExecutionContext(state.get('pipeline_memory'))
    
    # Get all structured inputs
    basic_info = context.get_step_output('basic_info', CompanyBasicInfo)
    reputation = context.get_step_output('reputation_check', ReputationAnalysis)
    social_metrics = context.get_step_output('social_analysis', SocialMediaMetrics)
    
    # Create comprehensive report
    final_report = CompanyReport(
        company_name=basic_info.name,
        research_date=datetime.now().strftime('%Y-%m-%d'),
        executive_summary=f"Analysis of {basic_info.name} in {basic_info.industry} industry",
        basic_info=basic_info,
        reputation=reputation,
        social_media={"main": social_metrics},
        overall_rating=calculate_overall_rating(reputation, social_metrics),
        recommendations=generate_recommendations(basic_info, reputation, social_metrics)
    )
    
    # Save as final output
    context.save_structured_output('final_report', final_report)
    
    return ('final_report', final_report.model_dump())
```

## Benefits of Structured Outputs

### 1. Type Safety
- Pydantic validation ensures data integrity
- IDE support with autocompletion
- Clear interface contracts between steps

### 2. Better Communication
- Steps can depend on specific data structures
- Reduced coupling through well-defined interfaces
- Clear documentation of what each step produces

### 3. Enhanced Learning
- Structured data enables pattern recognition
- Easier to extract insights from execution history
- Better memory management with typed data

### 4. Improved Debugging
- Clear visibility into step outputs
- Validation errors highlight data issues
- Structured logs and metrics

### 5. Flexibility
- Users can define custom schemas per use case
- Optional schemas for gradual adoption
- Backward compatibility with unstructured outputs

## Implementation Considerations

### 1. Schema Evolution
- Version schemas to handle changes
- Provide migration tools for schema updates
- Maintain backward compatibility

### 2. Performance
- Lazy validation to avoid unnecessary overhead
- Efficient serialization for large outputs
- Memory management for long-running pipelines

### 3. Error Handling
- Graceful fallbacks for validation failures
- Clear error messages for schema mismatches
- Recovery strategies for malformed outputs

### 4. Integration
- Seamless integration with existing TAgent tools
- Support for both structured and unstructured outputs
- Gradual migration path for existing code

This structured approach provides a powerful foundation for building sophisticated pipelines while maintaining flexibility and backward compatibility.
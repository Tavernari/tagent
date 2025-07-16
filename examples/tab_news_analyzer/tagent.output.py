from pydantic import BaseModel, Field

class TabNewsRecentsPostSummaries(BaseModel):
    """The final, structured output of the agent's work."""
    original_summary: str = Field(..., description="Original summary of the post")
    translated_summary: str = Field(..., description="Translated summary of the post")
    
output_schema = TabNewsRecentsPostSummaries
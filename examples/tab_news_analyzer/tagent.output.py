from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TabNewsRecentsPostSummary(BaseModel):
    title: str = Field(..., description="Title of the post")
    summary: str = Field(..., description="Summary of the post")
    url: str = Field(..., description="URL of the post")
    
class TabNewsRecentsPostSummaries(BaseModel):
    summaries: List[TabNewsRecentsPostSummary] = Field(..., description="List of post summaries")
    
output_schema = TabNewsRecentsPostSummaries
"""
Template Resolution Issue Validation Example

This example reproduces the template resolution issue where the load_web_page_tool
receives an unresolved template string like '{{task_1.output.0.url}}' instead of
the actual URL value from the previous task's output.

Based on the log analysis, the issue occurs when:
1. A search tool returns results with URLs
2. The next task tries to use {{task_1.output.0.url}} to reference the first URL
3. The template is not resolved and is passed literally to load_web_page_tool
4. This causes a DNS resolution error for the malformed URL

This validation case reproduces the exact scenario from the log.
"""

import requests
from typing import Dict, Optional, List
from pydantic import BaseModel, Field, HttpUrl
from bs4 import BeautifulSoup

from tagent import run_agent


# === Tool-Specific Pydantic Models ===

class SearchResult(BaseModel):
    """Represents a search result with URL and metadata."""
    title: str
    url: HttpUrl
    snippet: str


class SearchOutput(BaseModel):
    """Output contract for search tool."""
    results: List[SearchResult] = []
    error: Optional[str] = None


class WebPageInput(BaseModel):
    """Input contract for web page loading tool."""
    url: HttpUrl = Field(..., description="The URL to load content from")
    search_goal: str = Field(..., description="What to extract from the page")


class WebPageOutput(BaseModel):
    """Output contract for web page loading tool."""
    content: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None


# === Tools ===

def mock_reclame_aqui_search() -> SearchOutput:
    """
    Mock search tool that simulates the exact scenario from the log.
    Returns a Reclame Aqui search result for the company.
    """
    # This simulates the exact search result from the log
    mock_result = SearchResult(
        title="Lista de reclamações: 57.853.910 Daniel Cianelli Bull Gutierres ...",
        url="https://www.reclameaqui.com.br/empresa/57-853-910-daniel-cianelli-bull-gutierres/lista-reclamacoes/",
        snippet="Qual a reputação de 57.853.910 Daniel Cianelli Bull Gutierres? Sem reputação definida. Essa empresa ainda não possui 10 reclamações avaliadas para calcularmos a ..."
    )
    
    return SearchOutput(results=[mock_result])


def load_web_page_tool(args: WebPageInput) -> WebPageOutput:
    """
    Web page loading tool that demonstrates the template resolution issue.
    This tool will receive the unresolved template string and fail.
    """
    url_str = str(args.url)
    
    # Check if we received an unresolved template
    if "{{" in url_str and "}}" in url_str:
        return WebPageOutput(
            error=f"Template resolution failed: received unresolved template '{url_str}'",
            url=url_str
        )
    
    # Simulate actual web page loading
    try:
        response = requests.get(url_str, timeout=10)
        response.raise_for_status()
        
        # Simple content extraction
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.get_text(separator='\n', strip=True)
        
        return WebPageOutput(
            content=content[:1000],  # Truncate for demo
            url=url_str
        )
        
    except Exception as e:
        return WebPageOutput(
            error=f"Failed to load page: {str(e)}",
            url=url_str
        )


# === Final Output Model ===

class CompanyReputationAnalysis(BaseModel):
    """The final structured output of the reputation analysis."""
    company_name: str = Field(..., description="Name of the company analyzed")
    reputation_score: Optional[str] = Field(None, description="Company's reputation score")
    common_complaints: List[str] = Field(default_factory=list, description="List of common complaints")
    contact_info: Dict[str, str] = Field(default_factory=dict, description="Contact information found")
    analysis_status: str = Field(..., description="Status of the analysis (success/failed)")
    error_details: Optional[str] = Field(None, description="Error details if analysis failed")


# === Agent Execution ===

def run_template_resolution_validation():
    """
    Runs the validation test that reproduces the template resolution issue.
    """
    print("🔍 Running Template Resolution Issue Validation...")
    print("=" * 60)
    
    # This goal and task structure reproduces the exact scenario from the log
    goal = """You are a reputation analyst specializing in Reclame Aqui. Your goal is to assess the company '57.853.910 DANIEL CIANELLI BULL GUTIERRES'.

**Tasks:**
1. Use the search tool to find the company's profile on Reclame Aqui.
2. Use the web page loading tool to read the main profile page.
3. Summarize the company's reputation and findings.

The template resolution issue occurs when task 2 tries to use {{task_1.output.0.url}} to reference the URL from task 1's results."""
    
    result = run_agent(
        goal=goal,
        model="openrouter/google/gemini-2.5-flash",
        tools={
            "search_reclame_aqui": mock_reclame_aqui_search,
            "load_web_page_tool": load_web_page_tool
        },
        output_format=CompanyReputationAnalysis,
        max_iterations=1,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS:")
    print("=" * 60)
    
    print(f"Goal achieved: {result.goal_achieved}")
    print(f"Tasks completed: {result.completed_tasks}")
    print(f"Planning cycles: {result.planning_cycles}")
    
    if result.output:
        print(f"\nCompany: {result.output.company_name}")
        print(f"Analysis Status: {result.output.analysis_status}")
        if result.output.error_details:
            print(f"Error Details: {result.output.error_details}")
            
        # Check if we reproduced the template resolution issue
        if "template" in result.output.error_details.lower():
            print("✅ SUCCESS: Template resolution issue reproduced!")
            print("   The tool received an unresolved template string instead of the actual URL.")
        else:
            print("❌ ISSUE NOT REPRODUCED: The template resolution worked unexpectedly.")
    else:
        print("❌ No output generated")
    
    return result


if __name__ == "__main__":
    # Run the validation
    validation_result = run_template_resolution_validation()
    
    print("\n" + "=" * 60)
    print("🔧 DEBUGGING INFORMATION:")
    print("=" * 60)
    
    # Print additional debugging info
    print("This validation case tests the scenario where:")
    print("1. A search tool returns results with URLs")
    print("2. The next task uses {{task_1.output.0.url}} to reference the first URL")
    print("3. The template is not resolved properly")
    print("4. The load_web_page_tool receives the literal template string")
    print("5. This causes the tool to fail with a template resolution error")
    
    print("\nExpected behavior:")
    print("- The agent should resolve {{task_1.output.0.url}} to the actual URL")
    print("- The load_web_page_tool should receive the resolved URL")
    print("- The page content should be loaded successfully")
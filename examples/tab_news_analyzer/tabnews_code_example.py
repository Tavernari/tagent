from pdb import run
import requests
from typing import Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, Field, HttpUrl
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

from tagent import run_agent

# === Tool-Specific Pydantic Models (Input/Output Contracts) ===

# --- Models for extract_tabnews_articles ---

class Article(BaseModel):
    """Represents a single article from TabNews."""
    url: HttpUrl
    title: str
    publication_date: str

class ExtractTabnewsArticlesOutput(BaseModel):
    """Output contract for extract_tabnews_articles tool."""
    articles: List[Article] = []
    error: Optional[str] = None

# --- Models for load_and_process_url ---

class LoadUrlInput(BaseModel):
    """Input contract for the load_and_process_url tool."""
    url: HttpUrl = Field(..., description="The exact URL of the TabNews article to process.")

class LoadUrlOutput(BaseModel):
    """Output contract for the load_and_process_url tool."""
    content: Optional[str] = None
    error: Optional[str] = None

# --- Models for translate ---

class TranslateInput(BaseModel):
    """Input contract for the translate tool."""
    text: str = Field(..., description="The text to be translated.")
    target_language: str = Field(..., description="The target language for the translation (e.g., 'chinese', 'spanish').")

class TranslateOutput(BaseModel):
    """Output contract for the translate tool."""
    translation: Optional[str] = None
    original_text: Optional[str] = None
    target_language: Optional[str] = None
    error: Optional[str] = None


# === Tools (Now with simplified, flexible signatures) ===

def extract_tabnews_articles() -> ExtractTabnewsArticlesOutput:
    """
    Fetches recent news from TabNews, extracts the URLs, 
    titles, and publication dates, and returns them as a structured list.
    This tool does not require any arguments.
    """
    url = "https://www.tabnews.com.br/recentes/rss"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        articles_list = [
            Article(
                url=item.find('link').text,
                title=item.find('title').text,
                publication_date=item.find('pubDate').text
            ) for item in root.findall('.//item')
        ]
            
        return ExtractTabnewsArticlesOutput(articles=articles_list)

    except requests.exceptions.RequestException as e:
        return ExtractTabnewsArticlesOutput(error=f"Failed to fetch news: {e}")

def process_html_content(html_content: bytes) -> Optional[str]:
    """
    Parses HTML content to find the <main> tag and the first <div> within it,
    returning the cleaned text.
    """
    if not html_content:
        return None
    soup = BeautifulSoup(html_content, 'html.parser')
    main_tag = soup.find('main')
    if not main_tag:
        return None
    content_div = main_tag.find('div')
    if not content_div:
        return None
    text = content_div.get_text(separator='\n', strip=True)
    return "\n".join(line for line in text.splitlines() if line.strip())

def load_and_process_url(args: LoadUrlInput) -> LoadUrlOutput:
    """
    Fetches the content of a TabNews URL and extracts the main text content.
    """
    if "tabnews.com.br" not in str(args.url):
        return LoadUrlOutput(error="The URL must be from the tabnews.com.br domain.")

    try:
        response = requests.get(str(args.url), timeout=10)
        response.raise_for_status()

        main_text = process_html_content(response.content)

        if not main_text:
            return LoadUrlOutput(error="Failed to process the HTML content of the page.")

        return LoadUrlOutput(content=main_text)

    except requests.exceptions.RequestException as e:
        return LoadUrlOutput(error=f"Failed to fetch the URL: {e}")

def translate(args: TranslateInput) -> TranslateOutput:
    """

    Translates text to a target language using a direct LLM call.
    """
    try:
        import litellm
        messages = [
            {"role": "system", "content": "You are a professional translator. Provide accurate translations without additional commentary."},
            {"role": "user", "content": f"Translate the following text to {args.target_language}: {args.text}"}
        ]
        
        response = litellm.completion(
            model="openrouter/google/gemini-2.5-flash-lite-preview-06-17",
            messages=messages,
            temperature=0.3
        )
        
        translation = response.choices[0].message.content.strip()
        return TranslateOutput(
            translation=translation, 
            original_text=args.text, 
            target_language=args.target_language
        )
        
    except Exception as e:
        return TranslateOutput(error=f"Translation failed: {str(e)}")
    

# === Final Output Model ===

class TabNewsRecentsPostSummaries(BaseModel):
    """The final, structured output of the agent's work."""
    original_summary: str = Field(..., description="Original summary of the post")
    translated_summary: str = Field(..., description="Translated summary of the post")
    

# === Agent Execution ===

run_agent_response = run_agent(
    goal="Load articles from tabnews, select just one, then read the post content, summarize, and then translate to chinese. It is not required to load all articles.",
    model="openrouter/google/gemini-2.5-flash",
    tools={
        "extract_tabnews_articles": extract_tabnews_articles,
        "load_url_content": load_and_process_url,
        "translate": translate
    },
    output_format=TabNewsRecentsPostSummaries,
    max_iterations=2,
    verbose=False
)

# Access the output from the TaskBasedAgentResult
output = run_agent_response.output

print("--- TabNews Article ---\n")
print(f"Goal achieved: {run_agent_response.goal_achieved}")
print(f"Tasks completed: {run_agent_response.completed_tasks}")
print(f"Planning cycles: {run_agent_response.planning_cycles}")

if output:
    print("\nOriginal Summary:\n")
    print(output.original_summary)
    print("\nTranslated Summary:\n")
    print(output.translated_summary)
else:
    print("❌ No output generated")


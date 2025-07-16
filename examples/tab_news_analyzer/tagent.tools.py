import requests
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup


def extract_tabnews_articles(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Fetches a list of recent article titles and URLs from the TabNews RSS feed.
    This tool is the first step to get articles for processing.
    Keywords: TabNews, articles, list, fetch, extract, RSS.
    """
    url = "https://www.tabnews.com.br/recentes/rss"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        articles_list = [
            {
                "url": item.find('link').text,
                "title": item.find('title').text,
                "publication_date": item.find('pubDate').text
            }
            for item in root.findall('.//item')
        ]
            
        return ("articles", articles_list)

    except requests.exceptions.RequestException as e:
        return ("articles", f"Failed to fetch news: {e}")

def load_url_content(state: Dict[str, Any], args: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """
    Loads the full text content of a given URL. This tool is used to get the
    content of a specific article before summarizing or processing it.
    Keywords: load, URL, content, fetch, text.
    """
    url = args.get("url", "")
    if "tabnews.com.br" not in url:
        return ("url_content", {"error": "The URL must be from the tabnews.com.br domain"})

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        main_tag = soup.find('main')
        if not main_tag:
            return ("url_content", {"error": "Could not find the main content of the page."})
            
        text = main_tag.get_text(separator='\n', strip=True)
        cleaned_text = "\n".join(line for line in text.splitlines() if line.strip())

        return ("url_content", {"content": cleaned_text})

    except requests.exceptions.RequestException as e:
        return ("url_content", {"error": f"Failed to fetch the URL: {e}"})

def summarize_text(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Summarizes a given text using an LLM. This is useful for creating a short
    summary of a long article content.
    Keywords: summarize, text, summary, LLM.
    
    Args:
        state: Current agent state.
        args: Tool arguments with the following keys:
            - text (str): The text to be summarized.
    """
    import litellm

    text_to_summarize = args.get("text")
    if not text_to_summarize:
        return ("summary", {"error": "The 'text' argument is required."})

    try:
        prompt = f"Summarize the following text. Return only the summary, with no additional comments:\n\n{text_to_summarize}"
        messages = [{"role": "user", "content": prompt}]
        
        response = litellm.completion(
            model="gemini/gemini-pro", 
            messages=messages,
            temperature=0.2
        )
        
        summary = response.choices[0].message.content.strip()
        return ("summary", {"text": summary})

    except Exception as e:
        return ("summary", {"error": f"Failed to summarize text: {e}"})

def translate(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Translates a given text to a target language using an LLM.
    Keywords: translate, language, text.

    Args:
        state: Current agent state.
        args: Tool arguments with the following keys:
            - text (str): The text to be translated.
            - target_language (str): The language to translate to (e.g., "Chinese", "English").
    """
    import litellm

    text_to_translate = args.get("text")
    target_language = args.get("target_language")

    if not text_to_translate or not target_language:
        return ("translated_text", {"error": "The 'text' and 'target_language' arguments are required."})

    try:
        prompt = f"Translate the following text to {target_language}. Only return the translated text, with no additional comments or explanations:\n\n{text_to_translate}"
        
        messages = [{"role": "user", "content": prompt}]
        
        response = litellm.completion(
            model="gemini/gemini-pro", 
            messages=messages,
            temperature=0.1
        )
        
        translated_text = response.choices[0].message.content.strip()
        return ("translated_text", {"text": translated_text, "language": target_language})

    except Exception as e:
        return ("translated_text", {"error": f"Failed to translate text: {e}"})
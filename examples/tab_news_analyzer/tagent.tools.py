import requests
import xml.etree.ElementTree as ET

def extract_tabnews_articles(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Fetches recent news from TabNews, extracts the URLs, 
    titles, and publication dates, and returns them as a list of dictionaries.
    """
    url = "https://www.tabnews.com.br/recentes/rss"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for non-200 status codes
        
        root = ET.fromstring(response.content)
        
        articles_list = []
        
        for item in root.findall('.//item'):
            link = item.find('link').text
            title = item.find('title').text
            pub_date = item.find('pubDate').text
            
            articles_list.append({
                "url": link,
                "title": title,
                "publication_date": pub_date
            })
            
        return ("articles", articles_list)

    except requests.exceptions.RequestException as e:
        return ("articles", f"Failed to fetch news: {e}")


def load_url_content(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Fetches the content of a URL.
    
    Args:
        state: Current agent state
        args: Tool arguments 
            - url: URL to fetch content from
    
    Returns:
        Tuple with URL content
    """
    if "tabnews.com.br" not in args.get("url", ""):
        return ("url_content", {"error": "URL must be from TabNews"})

    url = args.get("url", "")

    response = requests.get(url)

    if response.status_code != 200:
        return ("url_content", {"error": "Failed to fetch URL content"})

    return ("url_content", response.content)
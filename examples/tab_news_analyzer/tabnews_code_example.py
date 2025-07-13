import requests
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

from tagent import run_agent

# === Tools ===


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

def process_html_content(html_content: bytes) -> Optional[str]:
    """
    Analisa o conteúdo HTML, encontra a tag <main> e o primeiro <div> dentro dela,
    e retorna o texto limpo.

    Args:
        html_content: O conteúdo HTML da página em bytes.

    Returns:
        Uma string com o texto extraído e limpo, ou None se não encontrar.
    """
    if not html_content:
        return None

    # 1. Inicia o BeautifulSoup para analisar o HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # 2. Encontra a tag <main>
    main_tag = soup.find('main')
    if not main_tag:
        print("Erro: A tag <main> não foi encontrada no HTML.")
        return None

    # 3. Encontra o primeiro <div> dentro da tag <main>
    # O método .find() retorna a primeira ocorrência que encontrar.
    content_div = main_tag.find('div')
    if not content_div:
        print("Erro: Nenhum <div> foi encontrado dentro da tag <main>.")
        return None
        
    # 4. Extrai todo o texto do div e seus filhos
    # O 'separator' ajuda a manter as quebras de linha entre os parágrafos
    text = content_div.get_text(separator='\n', strip=True)
    
    # 5. Limpeza final: remove linhas vazias extras para um resultado mais limpo
    cleaned_text = "\n".join(line for line in text.splitlines() if line.strip())

    return cleaned_text

def load_and_process_url(state: Dict[str, Any], args: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """
    Busca o conteúdo de uma URL do TabNews e extrai o conteúdo principal.
    
    Args:
        state: Estado atual do agente (não utilizado aqui, mas mantido por padrão).
        args: Argumentos da ferramenta, esperando uma chave "url".
    
    Returns:
        Uma tupla com o nome do campo a ser atualizado no estado e um dicionário
        contendo o 'content' ou um 'error'.
    """
    url = args.get("url", "")
    if "tabnews.com.br" not in url:
        return ("url_content", {"error": "A URL deve ser do domínio tabnews.com.br"})

    try:
        # Define um timeout para evitar que a requisição fique presa indefinidamente
        response = requests.get(url, timeout=10)
        # Lança uma exceção para códigos de erro HTTP (4xx ou 5xx)
        response.raise_for_status()

        # Usa nossa nova função para processar o conteúdo
        main_text = process_html_content(response.content)

        if not main_text:
            return ("url_content", {"error": "Falha ao processar o conteúdo HTML da página."})

        # Retorna o conteúdo processado com sucesso
        return ("url_content", {"content": main_text})

    except requests.exceptions.RequestException as e:
        # Captura erros de rede (DNS, conexão, timeout, etc.)
        return ("url_content", {"error": f"Falha ao buscar a URL: {e}"})

def translate(state: Dict[str, Any], args: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """
    Translates text from one language to another using direct LLM call.

    Args:
        state: State of the agent (not used here, but required by the interface)
        args: Arguments 
        - text: Text to translate
        - target_language: Target language for translation
    Returns:
        A tuple containing the name of the state variable to update and a dictionary with the translated text
    """
    text = args.get("text", "")
    target_language = args.get("target_language", "")
    
    if not text or not target_language:
        return ("translation", {"error": "Both text and target_language are required"})

    try:
        import litellm
        messages = [
            {"role": "system", "content": "You are a professional translator. Provide accurate translations without additional commentary."},
            {"role": "user", "content": f"Translate the following text to {target_language}: {text}"}
        ]
        
        response = litellm.completion(
            model="openrouter/google/gemini-2.5-flash-lite-preview-06-17",
            messages=messages,
            temperature=0.3
        )
        
        translation = response.choices[0].message.content.strip()
        return ("translation", {"translation": translation, "original_text": text, "target_language": target_language})
        
    except Exception as e:
        return ("translation", {"error": f"Translation failed: {str(e)}"})
    

# === Output Models ===

class TabNewsRecentsPostSummaries(BaseModel):
    original_summary: str = Field(..., description="Original summary of the post")
    translated_summary: str = Field(..., description="Translated summary of the post")
    

# === Simple Test ===

run_agent_response = run_agent(
    goal="Load articles from tabnews, select just one to load summarize and translate to chinese. It is not required to load all articles.",
    model="openrouter/google/gemini-2.5-flash-lite-preview-06-17",
    tools={
        "extract_tabnews_articles": extract_tabnews_articles,
        "load_url_content": load_and_process_url,
        "translate": translate
    },
    output_format=TabNewsRecentsPostSummaries,
    max_iterations=30,
    verbose=False
)

result = run_agent_response.get("result")

print("--- TabNews Article ---\n")
print("Original Summary:\n")
print(result.original_summary)
print("\nTranslated Summary:\n")
print(result.translated_summary)



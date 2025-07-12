

import json
from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel, Field
from tagent.agent import Store, run_agent

# --- 1. Definição do Modelo Pydantic para a Saída Final ---
# Este modelo garante que a saída do agente seja um JSON estruturado e validado.

class EnrichedCnpj(BaseModel):
    cnpj_consultado: str = Field(..., description="O CNPJ que foi consultado.")
    dados_cadastrais: Optional[Dict[str, Any]] = Field(None, description="Informações cadastrais da empresa.")
    instagram: Optional[Dict[str, Any]] = Field(None, description="Perfil do Instagram, se encontrado.")
    facebook: Optional[Dict[str, Any]] = Field(None, description="Página do Facebook, se encontrada.")
    contatos_telefonicos: List[Dict[str, str]] = Field([], description="Lista de telefones de contato.")
    resumo_enriquecimento: str = Field(..., description="Um resumo gerado por IA sobre os dados coletados.")

# --- 2. Definição das Ferramentas (Tools) Fakes ---
# Cada função é adaptada para o formato esperado pelo Store do TAgent:
# Recebe (state, args) e retorna uma tupla (key_to_update, value).

def busca_geral_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Realiza busca por dados cadastrais básicos de um CNPJ.
    
    Args:
        state: Estado atual do agente
        args: Dicionário com argumentos da ferramenta
            - cnpj (str): CNPJ a ser consultado no formato XX.XXX.XXX/XXXX-XX
            
    Returns:
        Tupla com ('dados_cadastrais', dados) onde dados contém:
        - razao_social: Razão social da empresa
        - nome_fantasia: Nome fantasia
        - situacao: Situação cadastral (ATIVA/INATIVA)
        - data_abertura: Data de abertura da empresa
        
    Example:
        args = {"cnpj": "00.000.000/0001-91"}
    """
    cnpj = args.get('cnpj')
    if not cnpj:
        return None
    
    # Dados Fakes
    dados = {
        "razao_social": "EMPRESA EXEMPLO LTDA",
        "nome_fantasia": "Empresa Exemplo",
        "situacao": "ATIVA",
        "data_abertura": "2010-05-20"
    }
    
    # Atualiza a chave 'dados_cadastrais' no estado do agente
    return ('dados_cadastrais', dados)

def buscar_instagram_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Busca perfil do Instagram associado a um CNPJ.
    
    Args:
        state: Estado atual do agente
        args: Dicionário com argumentos da ferramenta
            - cnpj (str): CNPJ a ser consultado no formato XX.XXX.XXX/XXXX-XX
            
    Returns:
        Tupla com ('instagram', perfil) onde perfil contém:
        - usuario: Nome de usuário do Instagram (@empresa)
        - url: URL do perfil
        - seguidores: Número de seguidores
        - verificado: Se a conta é verificada (boolean)
        
    Example:
        args = {"cnpj": "00.000.000/0001-91"}
    """
    cnpj = args.get('cnpj')
    if not cnpj:
        return None

    # Dados Fakes
    perfil_ig = {
        "usuario": "@empresa_exemplo",
        "url": "https://instagram.com/empresa_exemplo",
        "seguidores": 15000,
        "verificado": True
    }
    
    # Atualiza a chave 'instagram' no estado
    return ('instagram', perfil_ig)

def buscar_facebook_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Busca página do Facebook associada a um CNPJ.
    
    Args:
        state: Estado atual do agente
        args: Dicionário com argumentos da ferramenta
            - cnpj (str): CNPJ a ser consultado no formato XX.XXX.XXX/XXXX-XX
            
    Returns:
        Tupla com ('facebook', perfil) onde perfil contém:
        - nome_pagina: Nome da página no Facebook
        - url: URL da página
        - curtidas: Número de curtidas da página
        
    Example:
        args = {"cnpj": "00.000.000/0001-91"}
    """
    cnpj = args.get('cnpj')
    if not cnpj:
        return None

    # Dados Fakes
    perfil_fb = {
        "nome_pagina": "Empresa Exemplo Oficial",
        "url": "https://facebook.com/empresaexemplo",
        "curtidas": 42000
    }
    
    # Atualiza a chave 'facebook' no estado
    return ('facebook', perfil_fb)

def buscar_telefone_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """
    Busca telefones de contato associados a um CNPJ.
    
    Args:
        state: Estado atual do agente (usado para acessar resultados existentes)
        args: Dicionário com argumentos da ferramenta
            - cnpj (str): CNPJ a ser consultado no formato XX.XXX.XXX/XXXX-XX
            
    Returns:
        Tupla com ('results', telefones) onde telefones é uma lista contendo:
        - tipo: Tipo do telefone (Comercial, Suporte, etc.)
        - numero: Número de telefone formatado
        
    Example:
        args = {"cnpj": "00.000.000/0001-91"}
    """
    cnpj = args.get('cnpj')
    if not cnpj:
        return None

    # Dados Fakes
    telefones = [
        {"tipo": "Comercial", "numero": "+55 11 98765-4321"},
        {"tipo": "Suporte", "numero": "0800 123 4567"}
    ]
    
    # Adiciona os resultados à lista 'results' no estado
    results = state.get('results', [])
    results.extend(telefones)
    return ('results', results) # 'results' é uma chave padrão que o agente pode usar

# --- 3. Configuração e Execução do Agente ---

if __name__ == "__main__":
    # O CNPJ a ser enriquecido
    cnpj_alvo = "00.000.000/0001-91"
    
    # O objetivo claro para o agente
    agent_goal = f"Enriquecer o CNPJ {cnpj_alvo} buscando dados cadastrais, perfis de redes sociais (Instagram, Facebook) e contatos telefônicos. Ao final, consolidar tudo em um formato estruturado e gerar um resumo."

    # Dicionário registrando as ferramentas disponíveis
    agent_tools = {
        "busca_geral": busca_geral_tool,
        "buscar_instagram": buscar_instagram_tool,
        "buscar_facebook": buscar_facebook_tool,
        "buscar_telefone": buscar_telefone_tool,
    }

    print("--- Iniciando Agente de Enriquecimento de CNPJ ---")
    
    # Executa o loop do agente
    # O agente usará um LLM (configurado no `agent.py`) para decidir qual ferramenta chamar em cada passo.
    # O `output_format` garante que a saída final seja um objeto do tipo `EnrichedCnpj`.
    final_output = run_agent(
        goal=agent_goal,
        model="openrouter/google/gemma-3-27b-it", # Modelo que o agente usará para tomar decisões
        tools=agent_tools,
        output_format=EnrichedCnpj
    )

    print("\n--- Resultado Final do Agente ---")
    if final_output:
        # Verificar se há histórico de chat no resultado
        if isinstance(final_output, dict) and 'chat_summary' in final_output:
            print("\n" + final_output['chat_summary'])
            
            # Verificar o status da execução
            status = final_output.get('status', 'unknown')
            print(f"\n--- STATUS: {status.upper().replace('_', ' ')} ---")
            
            if final_output['result']:
                print("\n--- RESULTADO ESTRUTURADO ---")
                # O Pydantic model pode ser facilmente convertido para um JSON
                json_output = final_output['result'].model_dump_json(indent=4)
                print(json_output)
            elif final_output.get('raw_data'):
                print("\n--- DADOS COLETADOS (SEM FORMATAÇÃO) ---")
                # Mostrar dados brutos coletados
                raw_data = final_output['raw_data']
                collected_data = {k: v for k, v in raw_data.items() 
                                if k not in ['goal', 'achieved', 'used_tools'] and v}
                json_output = json.dumps(collected_data, indent=4, ensure_ascii=False)
                print(json_output)
                
                if final_output.get('error'):
                    print(f"\n⚠️  Aviso: {final_output['error']}")
            else:
                print(f"\nErro: {final_output.get('error', 'Resultado não disponível')}")
        else:
            # Resultado antigo sem chat
            json_output = final_output.model_dump_json(indent=4)
            print(json_output)
    else:
        print("O agente não conseguiu gerar uma saída final.")


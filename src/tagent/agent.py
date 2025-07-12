# tagent.py
# Implementação do TAgent baseado no blueprint do documento técnico.
# Integração com LiteLLM para chamadas reais a LLMs, aproveitando JSON Mode (ver https://docs.litellm.ai/docs/completion/json_mode).
# Requisitos: pip install pydantic litellm

from typing import Dict, Any, Tuple, Optional, Callable
from pydantic import BaseModel, ValidationError
import json  # Para parsing de JSON structured outputs
import litellm  # Para chamadas unificadas a LLMs

# Ative debug verbose para chamadas LLM (ver https://github.com/BerriAI/litellm/issues/4988)
litellm.log_raw_request_response = True

# === Modelos Pydantic para Estado e Structured Outputs ===
class AgentState(BaseModel):
    """Representa o estado do agente como um dicionário tipado."""
    data: Dict[str, Any] = {}

class StructuredResponse(BaseModel):
    """Schema para structured outputs gerados por LLMs."""
    action: str  # ex: "plan", "execute", "summarize", "evaluate"
    params: Dict[str, Any] = {}
    reasoning: str = ""

# === Classe Store (Redux-inspired) ===
class Store:
    def __init__(self, initial_state: Dict[str, Any]):
        self.state = AgentState(data=initial_state)
        self.tools: Dict[str, Callable] = {}  # Registro de ferramentas customizadas

    def register_tool(self, name: str, tool_func: Callable[[Dict[str, Any], Dict[str, Any]], Optional[Tuple[str, BaseModel]]]):
        """Registra uma ferramenta customizada como ação."""
        self.tools[name] = tool_func

    def dispatch(self, action_func: Callable[[Dict[str, Any]], Optional[Tuple[str, BaseModel]]]) -> None:
        """Despacha uma ação: chama a função, aplica reducer."""
        print("[INFO] Dispatching action...")
        result = action_func(self.state.data)
        if result:
            key, value = result
            self.state.data[key] = value
        print(f"[LOG] State updated: {self.state.data}")

# === Funções Auxiliares: Consulta a LLM com Structured Output via LiteLLM ===
def query_llm(prompt: str, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, max_retries: int = 3) -> StructuredResponse:
    """
    Consulta um LLM via LiteLLM e força structured output (JSON).
    Verifica suporte a response_format dinamicamente (ver https://docs.litellm.ai/docs/completion/json_mode).
    """
    # System prompt com few-shot example para melhorar outputs (inspirado em https://python.langchain.com/docs/how_to/debugging/)
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant designed to output JSON. Exemplo: {'action': 'execute', 'params': {'tool': 'fetch_data', 'args': {'query': 'clima'}}, 'reasoning': 'Preciso buscar dados.'}"
    }
    
    user_message = {
        "role": "user",
        "content": (
            f"{prompt}\n"
            "Ferramentas disponíveis: fetch_data (para buscar dados como clima).\n"
            "Responda SOMENTE com um JSON válido no formato: "
            "{'action': str (plan|execute|summarize|evaluate), 'params': dict, 'reasoning': str}."
            "Não adicione texto extra."
        )
    }
    
    messages = [system_message, user_message]
    
    # Verifica se o modelo suporta response_format (conforme docs)
    supported_params = litellm.get_supported_openai_params(model=model)
    response_format = {"type": "json_object"} if "response_format" in supported_params else None
    
    for attempt in range(max_retries):
        try:
            # Chamada via LiteLLM, passando api_key se fornecido
            response = litellm.completion(
                model=model,
                messages=messages,
                response_format=response_format,  # Ativa JSON mode se suportado
                temperature=0.0,  # Baixa temperatura para outputs determinísticos
                api_key=api_key,  # Passa api_key diretamente se fornecido
            )
            json_str = response.choices[0].message.content.strip()
            print(f"[RESPONSE] Raw LLM output: {json_str}")
            
            # Parse e valide com Pydantic
            return StructuredResponse.model_validate_json(json_str)
        
        except (litellm.AuthenticationError, litellm.APIError, litellm.ContextWindowExceededError, ValidationError, json.JSONDecodeError) as e:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise ValueError("Failed to get valid structured output after retries")
    
    raise ValueError("Max retries exceeded")

# === Ações Padrão (System Default Actions) ===
# Nota: Algumas ações usam query_llm para decisões inteligentes.

def plan_action(state: Dict[str, Any], model: str, api_key: Optional[str]) -> Optional[Tuple[str, BaseModel]]:
    """Gera um plano via structured output de LLM."""
    goal = state.get('goal', '')
    prompt = f"Baseado no goal: {goal}. Gere um plano."
    response = query_llm(prompt, model, api_key)
    if response.action == "plan":
        return ('plan', response.params)
    return None

def execute_action(state: Dict[str, Any], store: Store, model: str, api_key: Optional[str]) -> Optional[Tuple[str, BaseModel]]:
    """Executa uma ação baseada em structured output, possivelmente chamando tools."""
    prompt = f"Baseado no state: {state}. Decida o que executar."
    response = query_llm(prompt, model, api_key)
    print(f"[DECISION] Execute decision: {response}")
    if response.action == "execute":
        tool_name = response.params.get('tool')
        tool_args = response.params.get('args', {})
        if tool_name and tool_name in store.tools:
            print(f"[INFO] Calling tool: {tool_name} with args: {tool_args}")
            return store.tools[tool_name](state, tool_args)
        else:
            print(f"[ERROR] Tool not found: {tool_name}. Full response: {response}")
    return None

def summarize_action(state: Dict[str, Any], model: str, api_key: Optional[str]) -> Optional[Tuple[str, BaseModel]]:
    """Resume o contexto."""
    prompt = f"Baseado no state: {state}. Gere um resumo."
    response = query_llm(prompt, model, api_key)
    print(f"[DECISION] Summarize decision: {response}")
    if response.action == "summarize":
        summary = {"content": response.reasoning}  # Usa reasoning como base para resumo
        return ('summary', summary)
    return None

def goal_evaluation_action(state: Dict[str, Any], model: str, api_key: Optional[str]) -> Optional[Tuple[str, BaseModel]]:
    """Avalia se o goal foi alcançado via structured output."""
    prompt = f"Baseado no state: {state}. Avalie se o goal foi alcançado (retorne achieved: true/false em params apenas se houver results ou summary)."
    response = query_llm(prompt, model, api_key)
    print(f"[DECISION] Evaluation decision: {response}")
    achieved = bool(response.params.get('achieved', False))  # Garante booleano
    return ('achieved', achieved)

# === Loop Principal ===
def run_agent(goal: str, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, max_iterations: int = 10) -> Dict[str, Any]:
    """Executa o loop principal do agente."""
    store = Store({'goal': goal, 'results': []})
    
    # Registra uma ferramenta de exemplo (simula chamada a API)
    def fetch_data_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, BaseModel]]:
        query = args.get('query', '')
        # Simula chamada (em produção, integre API real)
        data = {"data": f"Resultados para {query}: simulado via ferramenta"}
        results = state.get('results', []) + [data]
        print(f"[INFO] Tool fetch_data executed with query: {query}")
        return ('results', results)
    
    store.register_tool('fetch_data', fetch_data_tool)
    
    iteration = 0
    while not store.state.data.get('achieved', False) and iteration < max_iterations:
        iteration += 1
        print(f"[LOOP] Iteration {iteration}. Current state: {store.state.data}")
        
        # Consulta LLM para decidir a próxima ação (structured output)
        prompt = f"Estado atual: {store.state.data}. Decida a próxima ação para o goal '{goal}'."
        decision = query_llm(prompt, model, api_key)
        print(f"[DECISION] LLM decided: {decision}")
        
        # Dispatch baseado na decisão do LLM
        if decision.action == "plan":
            store.dispatch(lambda state: plan_action(state, model, api_key))
        elif decision.action == "execute":
            result = execute_action(store.state.data, store, model, api_key)
            if result:
                key, value = result
                store.state.data[key] = value
                print(f"[LOG] Execute updated: {store.state.data}")
        elif decision.action == "summarize":
            store.dispatch(lambda state: summarize_action(state, model, api_key))
        elif decision.action == "evaluate":
            store.dispatch(lambda state: goal_evaluation_action(state, model, api_key))
        else:
            print(f"[WARNING] Unknown action: {decision.action}")
        
        # Sempre avalia no final da iteração
        store.dispatch(lambda state: goal_evaluation_action(state, model, api_key))
    
    if store.state.data.get('achieved', False):
        print("[SUCCESS] Goal achieved!")
    else:
        print("[WARNING] Max iterations reached.")
    
    return store.state.data

# === Exemplo de Uso ===
if __name__ == "__main__":
    final_state = run_agent("Criar um relatório sobre clima", model="ollama/gemma3")
    print("\nFinal State:", final_state)
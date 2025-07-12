# tagent.py
# Implementação do TAgent baseado no blueprint do documento técnico.
# Integração com LiteLLM para chamadas reais a LLMs, aproveitando JSON Mode (ver https://docs.litellm.ai/docs/completion/json_mode).
# Requisitos: pip install pydantic litellm

from typing import Dict, Any, Tuple, Optional, Callable, Type, List
from pydantic import BaseModel, ValidationError, Field
import json  # Para parsing de JSON structured outputs
import litellm  # Para chamadas unificadas a LLMs
import inspect  # Para introspecção de funções

# Ative debug verbose para chamadas LLM (ver https://github.com/BerriAI/litellm/issues/4988)
litellm.log_raw_request_response = False

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
        self.conversation_history: List[Dict[str, str]] = []  # Histórico da conversa

    def register_tool(self, name: str, tool_func: Callable[[Dict[str, Any], Dict[str, Any]], Optional[Tuple[str, BaseModel]]]):
        """Registra uma ferramenta customizada como ação."""
        self.tools[name] = tool_func

    def add_to_conversation(self, role: str, content: str) -> None:
        """Adiciona mensagem ao histórico da conversa."""
        self.conversation_history.append({"role": role, "content": content})
    
    def add_assistant_response(self, response: StructuredResponse) -> None:
        """Adiciona resposta do assistente ao histórico de forma estruturada."""
        formatted_response = f"Action: {response.action}\nReasoning: {response.reasoning}\nParams: {response.params}"
        self.add_to_conversation("assistant", formatted_response)

    def dispatch(self, action_func: Callable[[Dict[str, Any]], Optional[Tuple[str, BaseModel]]]) -> None:
        """Despacha uma ação: chama a função, aplica reducer."""
        print("[INFO] Dispatching action...")
        result = action_func(self.state.data)
        if result:
            key, value = result
            self.state.data[key] = value
        print(f"[LOG] State updated: {self.state.data}")

# === Funções Auxiliares: Introspecção de Tools e Consulta a LLM ===

def get_tool_documentation(tools: Dict[str, Callable]) -> str:
    """
    Extrai documentação das ferramentas registradas incluindo docstrings e assinaturas.
    
    Args:
        tools: Dicionário de ferramentas registradas
        
    Returns:
        String formatada com documentação das ferramentas
    """
    if not tools:
        return ""
    
    tool_docs = []
    
    for tool_name, tool_func in tools.items():
        # Extrair assinatura da função
        try:
            sig = inspect.signature(tool_func)
            signature = f"{tool_name}{sig}"
        except (ValueError, TypeError):
            signature = f"{tool_name}(state, args)"
        
        # Extrair docstring
        docstring = inspect.getdoc(tool_func)
        if not docstring:
            docstring = "No documentation available"
        
        tool_doc = f"- {signature}: {docstring}"
        tool_docs.append(tool_doc)
    
    return "Available tools:\n" + "\n".join(tool_docs) + "\n"

def detect_action_loop(recent_actions: List[str], max_recent: int = 3) -> bool:
    """Detecta se o agente está em loop de ações repetidas."""
    if len(recent_actions) < max_recent:
        return False
    
    # Verificar se as últimas 3 ações são iguais
    last_actions = recent_actions[-max_recent:]
    return len(set(last_actions)) == 1

def format_conversation_as_chat(conversation_history: List[Dict[str, str]]) -> str:
    """
    Formata o histórico de conversação como um chat legível.
    
    Args:
        conversation_history: Lista de mensagens da conversa
        
    Returns:
        String formatada como chat
    """
    chat_lines = []
    chat_lines.append("=== HISTÓRICO DA CONVERSA ===\n")
    
    for i, message in enumerate(conversation_history, 1):
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        if role == 'user':
            chat_lines.append(f"👤 USER [{i}]:")
            chat_lines.append(f"   {content}\n")
        elif role == 'assistant':
            chat_lines.append(f"🤖 ASSISTANT [{i}]:")
            chat_lines.append(f"   {content}\n")
        else:
            chat_lines.append(f"📝 {role.upper()} [{i}]:")
            chat_lines.append(f"   {content}\n")
    
    chat_lines.append("=== FIM DO HISTÓRICO ===")
    return "\n".join(chat_lines)

# === Funções Auxiliares: Consulta a LLM com Structured Output via LiteLLM ===
def query_llm(prompt: str, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, max_retries: int = 3, tools: Optional[Dict[str, Callable]] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> StructuredResponse:
    """
    Consulta um LLM via LiteLLM e força structured output (JSON).
    Verifica suporte a response_format dinamicamente (ver https://docs.litellm.ai/docs/completion/json/mode).
    """
    # System prompt com few-shot example para melhorar outputs (inspirado em https://python.langchain.com/docs/how_to/debugging/)
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant designed to output JSON. Example: {'action': 'execute', 'params': {'tool': 'tool_name', 'args': {'parameter': 'value'}}, 'reasoning': 'Reason to execute the action.'}"
    }
    
    # Usar documentação detalhada das ferramentas se disponível
    available_tools = ""
    if tools:
        available_tools = get_tool_documentation(tools)

    user_message = {
        "role": "user",
        "content": (
            f"{prompt}\n\n"
            f"{available_tools}"
            "When using 'execute' action, choose the most appropriate tool based on its documentation. "
            "Ensure 'params' contains 'tool' (tool name) and 'args' (parameters matching the tool's signature).\n"
            "Respond ONLY with a valid JSON in the format: "
            "{'action': str (plan|execute|summarize|evaluate), 'params': dict, 'reasoning': str}."
            "Do not add extra text."
        )
    }

    
    # Construir mensagens incluindo histórico de conversação
    messages = [system_message]
    
    # Adicionar histórico de conversação se disponível
    if conversation_history:
        messages.extend(conversation_history)
    
    # Adicionar mensagem atual do usuário
    messages.append(user_message)
    
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

def query_llm_for_model(prompt: str, model: str, output_model: Type[BaseModel], api_key: Optional[str] = None, max_retries: int = 3) -> BaseModel:
    """
    Queries an LLM and enforces the output to conform to a specific Pydantic model.
    Improved with few-shot examples and error feedback in retries, inspired by [github.com](https://github.com/zby/LLMEasyTools) for Pydantic-structured outputs.
    Uses LiteLLM's JSON mode per [docs.litellm.ai](https://docs.litellm.ai/docs/reasoning_content).
    """
    # Generate a dummy example based on the schema (to guide the LLM)
    schema = output_model.model_json_schema()
    example_data = {field: f"example_{field}" for field in schema.get('properties', {})}
    example_json = json.dumps(example_data)

    error_feedback = ""  # Will accumulate errors for retries
    for attempt in range(max_retries):
        system_message = {
            "role": "system",
            "content": (
                f"You are a helpful assistant designed to output JSON conforming to the following schema: {json.dumps(schema)}.\n"
                f"Example output: {example_json}.\n"
                "Ensure ALL required fields are filled. Do not output empty objects."
            )
        }
        
        user_message = {
            "role": "user",
            "content": (
                f"{prompt}\n"
                f"Extract and format data from the state. {error_feedback}\n"
                "Respond ONLY with a valid JSON object matching the schema. No extra text."
            )
        }
        
        messages = [system_message, user_message]
        
        supported_params = litellm.get_supported_openai_params(model=model)
        response_format = {"type": "json_object"} if "response_format" in supported_params else None
        
        try:
            # Add model_kwargs for extra control, per [api.python.langchain.com](https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.litellm.ChatLiteLLM.html)
            response = litellm.completion(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=0.0,
                api_key=api_key,
                model_kwargs={"strict": True} if "strict" in supported_params else {},  # Enforce strict mode if supported
            )
            json_str = response.choices[0].message.content.strip()
            print(f"[RESPONSE] Raw LLM output for model query: {json_str}")
            
            return output_model.model_validate_json(json_str)
        
        except (litellm.AuthenticationError, litellm.APIError, litellm.ContextWindowExceededError, ValidationError, json.JSONDecodeError) as e:
            print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {e}")
            error_feedback = f"Previous output was invalid: {str(e)}. Correct it by filling all required fields like {list(schema['required'])}."
            if attempt == max_retries - 1:
                raise ValueError("Failed to get valid structured output after retries")
    
    raise ValueError("Max retries exceeded")

# === Ações Padrão (System Default Actions) ===
# (Remaining actions unchanged for brevity; they already use query_llm effectively)

# Nota: Algumas ações usam query_llm para decisões inteligentes.

def plan_action(state: Dict[str, Any], model: str, api_key: Optional[str], tools: Optional[Dict[str, Callable]] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[Tuple[str, BaseModel]]:
    """Generates a plan via LLM structured output."""
    goal = state.get('goal', '')
    used_tools = state.get('used_tools', [])
    available_tools = list(tools.keys()) if tools else []
    unused_tools = [t for t in available_tools if t not in used_tools]
    
    prompt = (
        f"Goal: {goal}\n"
        f"Current progress: {state}\n"
        f"Used tools: {used_tools}\n"
        f"Unused tools: {unused_tools}\n"
        "The current approach may not be working. Generate a new strategic plan. "
        "Consider: 1) What data is still missing? 2) What tools haven't been tried? "
        "3) What alternative approaches could work? 4) Should we try different parameters?"
    )
    response = query_llm(prompt, model, api_key, tools=tools, conversation_history=conversation_history)
    if response.action == "plan":
        return ('plan', response.params)
    return None


def summarize_action(state: Dict[str, Any], model: str, api_key: Optional[str], tools: Optional[Dict[str, Callable]] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[Tuple[str, BaseModel]]:
    """Summarizes the context."""
    prompt = f"Based on the state: {state}. Generate a summary."
    response = query_llm(prompt, model, api_key, tools=tools, conversation_history=conversation_history)
    print(f"[DECISION] Summarize decision: {response}")
    if response.action == "summarize":
        summary = {"content": response.reasoning}  # Use reasoning as the basis for the summary
        return ('summary', summary)
    return None

def goal_evaluation_action(state: Dict[str, Any], model: str, api_key: Optional[str], tools: Optional[Dict[str, Callable]] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[Tuple[str, BaseModel]]:
    """Evaluates if the goal has been achieved via structured output."""
    goal = state.get('goal', '')
    prompt = f"Based on the current state: {state} and the goal: '{goal}'. Evaluate if the goal has been sufficiently achieved. Consider the data collected and whether it meets the requirements."
    response = query_llm(prompt, model, api_key, tools=tools, conversation_history=conversation_history)
    print(f"[DECISION] Evaluation decision: {response}")
    achieved = bool(response.params.get('achieved', False))  # Ensure boolean
    return ('achieved', achieved)

def format_output_action(state: Dict[str, Any], model: str, api_key: Optional[str], output_format: Type[BaseModel]) -> Optional[Tuple[str, BaseModel]]:
    """Formats the final output according to the specified Pydantic model."""
    goal = state.get('goal', '')
    prompt = (
        f"Based on the final state: {state} and the original goal: '{goal}'. "
        "Extract and format all relevant data collected during the goal execution. "
        "Create appropriate summaries and ensure all required fields are filled according to the output schema."
    )
    formatted_output = query_llm_for_model(prompt, model, output_format, api_key)
    return ('final_output', formatted_output)


# === Loop Principal ===
def run_agent(goal: str, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, max_iterations: int = 10, tools: Optional[Dict[str, Callable]] = None, output_format: Optional[Type[BaseModel]] = None) -> Optional[BaseModel]:
    """
    Runs the main agent loop.

    Args:
        goal: The main objective for the agent.
        model: The LLM model to use.
        api_key: The API key for the LLM service.
        max_iterations: The maximum number of iterations.
        tools: A dictionary of custom tools to register with the agent.
        output_format: The Pydantic model for the final output.

    Returns:
        An instance of the `output_format` model, or None if no output is generated.
    """
    store = Store({'goal': goal, 'results': [], 'used_tools': []})
    
    # Sistema de proteção contra loops infinitos
    consecutive_failures = 0
    max_consecutive_failures = 5
    last_data_count = 0
    
    # Sistema de detecção de loops de ações
    recent_actions = []
    max_recent_actions = 3

    # Register tools if provided
    if tools:
        for name, tool_func in tools.items():
            store.register_tool(name, tool_func)

    iteration = 0
    while not store.state.data.get('achieved', False) and iteration < max_iterations and consecutive_failures < max_consecutive_failures:
        iteration += 1
        print(f"[LOOP] Iteration {iteration}. Current state: {store.state.data}")

        # Verificar se houve progresso real (reset contador de falhas)
        data_keys = [k for k, v in store.state.data.items() if k not in ['goal', 'achieved', 'used_tools'] and v]
        current_data_count = len(data_keys)
        
        if current_data_count > last_data_count:
            print(f"[PROGRESS] Data items increased from {last_data_count} to {current_data_count} - resetting failure counter")
            consecutive_failures = 0
            last_data_count = current_data_count
        
        progress_summary = f"Progress: {current_data_count} data items collected"
        
        used_tools = store.state.data.get('used_tools', [])
        unused_tools = [t for t in store.tools.keys() if t not in used_tools]
        
        # Detectar loop de ações e ajustar estratégia
        action_loop_detected = detect_action_loop(recent_actions, max_recent_actions)
        strategy_hint = ""
        
        if action_loop_detected:
            last_action = recent_actions[-1] if recent_actions else "unknown"
            print(f"[STRATEGY] Action loop detected: repeating '{last_action}' - suggesting strategy change")
            
            if last_action == "evaluate" and unused_tools:
                strategy_hint = f"IMPORTANT: You've been stuck evaluating. Try using unused tools first: {unused_tools}. "
            elif last_action == "evaluate" and not unused_tools:
                strategy_hint = "IMPORTANT: You've been stuck evaluating. Try 'plan' to reconsider strategy or 'execute' with different parameters. "
            elif unused_tools:
                strategy_hint = f"IMPORTANT: Break the pattern! Try unused tools: {unused_tools} or use 'plan' to rethink approach. "
            else:
                strategy_hint = "IMPORTANT: Break the pattern! Try 'plan' to develop new strategy or different parameters. "
        
        prompt = (
            f"Goal: {goal}\n"
            f"Current state: {store.state.data}\n"
            f"{progress_summary}\n"
            f"Used tools: {used_tools}\n"
            f"Unused tools: {unused_tools}\n"
            f"{strategy_hint}"
            "For 'execute' action, prefer UNUSED tools to gather different types of data. "
            "If all tools have been used and sufficient data collected, use 'evaluate'. "
            "Available actions: plan, execute, summarize, evaluate"
        )
        # Adicionar prompt atual ao histórico
        store.add_to_conversation("user", prompt)
        
        decision = query_llm(prompt, model, api_key, tools=store.tools, conversation_history=store.conversation_history[:-1])  # Excluir a última mensagem para evitar duplicação
        print(f"[DECISION] LLM decided: {decision}")
        
        # Rastrear ações recentes para detectar loops
        recent_actions.append(decision.action)
        if len(recent_actions) > max_recent_actions:
            recent_actions.pop(0)  # Manter apenas as últimas ações
        
        # Adicionar resposta do assistente ao histórico
        store.add_assistant_response(decision)

        # Dispatch based on LLM decision
        if decision.action == "plan":
            store.dispatch(lambda state: plan_action(state, model, api_key, tools=store.tools, conversation_history=store.conversation_history))
        elif decision.action == "execute":
            # Extract tool and args from the main decision
            tool_name = decision.params.get('tool')
            tool_args = decision.params.get('args', {})
            if tool_name and tool_name in store.tools:
                print(f"[INFO] Calling tool: {tool_name} with args: {tool_args}")
                result = store.tools[tool_name](store.state.data, tool_args)
                if result:
                    key, value = result
                    store.state.data[key] = value
                    # Track used tools
                    used_tools = store.state.data.get('used_tools', [])
                    if tool_name not in used_tools:
                        used_tools.append(tool_name)
                        store.state.data['used_tools'] = used_tools
                    print(f"[LOG] Execute updated: {store.state.data}")
            else:
                print(f"[ERROR] Tool not found: {tool_name}. Available tools: {list(store.tools.keys())}")
        elif decision.action == "summarize":
            store.dispatch(lambda state: summarize_action(state, model, api_key, tools=store.tools, conversation_history=store.conversation_history))
        elif decision.action == "evaluate":
            # Armazenar estado anterior para detectar mudança
            previous_achieved = store.state.data.get('achieved', False)
            store.dispatch(lambda state: goal_evaluation_action(state, model, api_key, tools=store.tools, conversation_history=store.conversation_history))
            
            # Se evaluation ainda retorna False, incrementar contador de falhas
            current_achieved = store.state.data.get('achieved', False)
            if not current_achieved and not previous_achieved:
                consecutive_failures += 1
                print(f"[FAILURE] Evaluator failed {consecutive_failures}/{max_consecutive_failures} times consecutively")
                
                # Forçar conclusão se muitas falhas consecutivas com dados suficientes
                if consecutive_failures >= max_consecutive_failures and current_data_count >= 3:
                    print(f"[FORCE] Forcing completion due to {consecutive_failures} consecutive failures with {current_data_count} data items")
                    store.state.data['achieved'] = True
        else:
            print(f"[WARNING] Unknown action: {decision.action}")
            # If unknown action, evaluate to potentially break the loop
            store.dispatch(lambda state: goal_evaluation_action(state, model, api_key, tools=store.tools, conversation_history=store.conversation_history))

    if store.state.data.get('achieved', False):
        print("[SUCCESS] Goal achieved!")
        if output_format:
            print("[INFO] Formatting final output...")
            try:
                store.dispatch(lambda state: format_output_action(state, model, api_key, output_format))
                
                # Adicionar histórico de conversação ao resultado final
                final_result = store.state.data.get('final_output')
                if final_result:
                    # Criar resultado com histórico de chat
                    final_result_with_chat = {
                        'result': final_result,
                        'conversation_history': store.conversation_history,
                        'chat_summary': format_conversation_as_chat(store.conversation_history),
                        'status': 'completed_with_formatting'
                    }
                    return final_result_with_chat
                return store.state.data.get('final_output')
                
            except Exception as e:
                print(f"[ERROR] Failed to format final output: {e}")
                print("[INFO] Returning raw collected data instead")
                
                # Retornar dados coletados mesmo sem formatação
                return {
                    'result': None,
                    'raw_data': store.state.data,
                    'conversation_history': store.conversation_history,
                    'chat_summary': format_conversation_as_chat(store.conversation_history),
                    'status': 'completed_without_formatting',
                    'error': f"Formatting failed: {str(e)}"
                }
    else:
        # Determinar o motivo da parada
        if consecutive_failures >= max_consecutive_failures:
            error_msg = f"Stopped due to {consecutive_failures} consecutive evaluator failures"
            print(f"[WARNING] {error_msg}")
        elif iteration >= max_iterations:
            error_msg = "Max iterations reached"
            print(f"[WARNING] {error_msg}")
        else:
            error_msg = "Unknown termination reason"
            print(f"[WARNING] {error_msg}")
            
        # Retornar histórico mesmo se não completou
        return {
            'result': None,
            'conversation_history': store.conversation_history,
            'chat_summary': format_conversation_as_chat(store.conversation_history),
            'error': error_msg,
            'final_state': store.state.data
        }

    return None

# === Example Usage ===
if __name__ == "__main__":
    import time

    # Define a fake tool to fetch weather data with a delay
    def fetch_weather_tool(state: Dict[str, Any], args: Dict[str, Any]) -> Optional[Tuple[str, BaseModel]]:
        location = args.get('location', 'default')
        print(f"[INFO] Fetching weather for {location}...")
        time.sleep(3)
        # Simulated weather data
        weather_data = {"location": location, "temperature": "25°C", "condition": "Sunny"}
        results = state.get('results', []) + [weather_data]
        print(f"[INFO] Weather data fetched for {location}.")
        return ('results', results)

    # Create a dictionary of tools to register
    agent_tools = {
        "fetch_weather": fetch_weather_tool
    }

    # Define the desired output format
    class WeatherReport(BaseModel):
        location: str = Field(..., description="The location of the weather report.")
        temperature: str = Field(..., description="The temperature in Celsius.")
        condition: str = Field(..., description="The weather condition.")
        summary: str = Field(..., description="A summary of the weather report.")

    # Create the agent and pass the tools and output format
    agent_goal = "Create a weather report for London."
    final_state = run_agent(
        goal=agent_goal,
        model="ollama/gemma3",
        tools=agent_tools,
        output_format=WeatherReport
    )
    print("\nFinal State:", final_state)
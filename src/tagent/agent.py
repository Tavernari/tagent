# tagent.py
# Implementação do TAgent baseado no blueprint do documento técnico.
# Integração com LiteLLM para chamadas reais a LLMs, aproveitando JSON Mode (ver https://docs.litellm.ai/docs/completion/json_mode).
# Requisitos: pip install pydantic litellm

from typing import Dict, Any, Tuple, Optional, Callable, Type
from pydantic import BaseModel, ValidationError, Field
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
def query_llm(prompt: str, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, max_retries: int = 3, tools: Optional[Dict[str, Callable]] = None) -> StructuredResponse:
    """
    Consulta um LLM via LiteLLM e força structured output (JSON).
    Verifica suporte a response_format dinamicamente (ver https://docs.litellm.ai/docs/completion/json/mode).
    """
    # System prompt com few-shot example para melhorar outputs (inspirado em https://python.langchain.com/docs/how_to/debugging/)
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant designed to output JSON. Example: {'action': 'execute', 'params': {'tool': 'tool_name', 'args': {'parameter': 'value'}}, 'reasoning': 'Reason to execute the action.'}"
    }
    
    available_tools = ""
    if tools:
        tool_names = ", ".join(tools.keys())
        available_tools = f"Available tools: {tool_names} (to execute external actions).\n"

    user_message = {
        "role": "user",
        "content": (
            f"{prompt}\n"
            f"{available_tools}"
            "Respond ONLY with a valid JSON in the format: "
            "{'action': str (plan|execute|summarize|evaluate), 'params': dict, 'reasoning': str}."
            "Do not add extra text."
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

def plan_action(state: Dict[str, Any], model: str, api_key: Optional[str], tools: Optional[Dict[str, Callable]] = None) -> Optional[Tuple[str, BaseModel]]:
    """Generates a plan via LLM structured output."""
    goal = state.get('goal', '')
    prompt = f"Based on the goal: '{goal}'. Generate a plan."
    response = query_llm(prompt, model, api_key, tools=tools)
    if response.action == "plan":
        return ('plan', response.params)
    return None

def execute_action(state: Dict[str, Any], store: Store, model: str, api_key: Optional[str], tools: Optional[Dict[str, Callable]] = None) -> Optional[Tuple[str, BaseModel]]:
    """Executes an action based on structured output, possibly calling tools."""
    prompt = f"Based on the state: {state}. Decide what to execute."
    response = query_llm(prompt, model, api_key, tools=tools)
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

def summarize_action(state: Dict[str, Any], model: str, api_key: Optional[str], tools: Optional[Dict[str, Callable]] = None) -> Optional[Tuple[str, BaseModel]]:
    """Summarizes the context."""
    prompt = f"Based on the state: {state}. Generate a summary."
    response = query_llm(prompt, model, api_key, tools=tools)
    print(f"[DECISION] Summarize decision: {response}")
    if response.action == "summarize":
        summary = {"content": response.reasoning}  # Use reasoning as the basis for the summary
        return ('summary', summary)
    return None

def goal_evaluation_action(state: Dict[str, Any], model: str, api_key: Optional[str], tools: Optional[Dict[str, Callable]] = None) -> Optional[Tuple[str, BaseModel]]:
    """Evaluates if the goal has been achieved via structured output."""
    prompt = f"Based on the state: {state}. Evaluate if the goal has been achieved (return achieved: true/false in params only if there are results or a summary)."
    response = query_llm(prompt, model, api_key, tools=tools)
    print(f"[DECISION] Evaluation decision: {response}")
    achieved = bool(response.params.get('achieved', False))  # Ensure boolean
    return ('achieved', achieved)

def format_output_action(state: Dict[str, Any], model: str, api_key: Optional[str], output_format: Type[BaseModel]) -> Optional[Tuple[str, BaseModel]]:
    """Formats the final output according to the specified Pydantic model."""
    # Improved prompt to guide extraction from state (e.g., results)
    prompt = (
        f"Based on the final state: {state}. "
        "Extract weather data from 'results' (e.g., location, temperature, condition) and create a summary. "
        "Format the output according to the schema."
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
    store = Store({'goal': goal, 'results': []})

    # Register tools if provided
    if tools:
        for name, tool_func in tools.items():
            store.register_tool(name, tool_func)

    iteration = 0
    while not store.state.data.get('achieved', False) and iteration < max_iterations:
        iteration += 1
        print(f"[LOOP] Iteration {iteration}. Current state: {store.state.data}")

        # Query LLM to decide the next action (structured output)
        prompt = f"Current state: {store.state.data}. Decide the next action for the goal '{goal}'."
        decision = query_llm(prompt, model, api_key, tools=store.tools)
        print(f"[DECISION] LLM decided: {decision}")

        # Dispatch based on LLM decision
        if decision.action == "plan":
            store.dispatch(lambda state: plan_action(state, model, api_key, tools=store.tools))
        elif decision.action == "execute":
            result = execute_action(store.state.data, store, model, api_key, tools=store.tools)
            if result:
                key, value = result
                store.state.data[key] = value
                print(f"[LOG] Execute updated: {store.state.data}")
        elif decision.action == "summarize":
            store.dispatch(lambda state: summarize_action(state, model, api_key, tools=store.tools))
        elif decision.action == "evaluate":
            store.dispatch(lambda state: goal_evaluation_action(state, model, api_key, tools=store.tools))
        else:
            print(f"[WARNING] Unknown action: {decision.action}")

        # Always evaluate at the end of the iteration
        store.dispatch(lambda state: goal_evaluation_action(state, model, api_key, tools=store.tools))

    if store.state.data.get('achieved', False):
        print("[SUCCESS] Goal achieved!")
        if output_format:
            print("[INFO] Formatting final output...")
            store.dispatch(lambda state: format_output_action(state, model, api_key, output_format))
            return store.state.data.get('final_output')
    else:
        print("[WARNING] Max iterations reached.")

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
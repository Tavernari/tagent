# Documento Técnico: Arquitetura do TAgent

## Introdução
O TAgent é um agente autônomo projetado para tarefas de raciocínio e execução, inspirado na arquitetura Redux clássica, mas simplificado para cenários de IA. Ele gerencia um estado centralizado atualizado por ações, permitindo loops de decisão proativos. Para maximizar compatibilidade com diversos modelos de Large Language Models (LLMs), o TAgent evita dependência em "function calls" nativos, optando por **Structured Outputs**. Isso permite que o agente use saídas estruturadas (ex: JSON) geradas por LLMs via prompting, interpretadas em um loop controlado pelo código.

O TAgent é ideal para aplicações como automação de tarefas, assistentes de IA ou workflows autônomos, onde o agente planeja, executa e avalia ações até alcançar um objetivo (goal).

## Características Principais
- **Baseado em Redux Simplificado**: Usa estado, ações, reducer e store para gerenciamento determinístico e previsível.
- **Autonomia via Loop**: Um loop principal decide e executa ações com base no estado, simulando raciocínio agente (ex: ReAct-like).
- **Integração com Structured Outputs**: Em vez de function calls, consultas a LLMs geram saídas estruturadas (ex: JSON com schema fixo), que são parseadas e despachadas como ações. Isso torna o sistema compatível com LLMs sem suporte a tool calling.
- **Estado como Dicionário**: Estado é um dicionário `{str: BaseModel}` para tipagem e validação (usando bibliotecas como Pydantic).
- **Ações como Funções**: Ações retornam atualizações ao estado (ex: tuple `(key: str, value: BaseModel)`).
- **Extensibilidade**: Suporte a ações customizadas (ferramentas) e integração com LLMs para decisões inteligentes.
- **Opcionais Mínimos**: Sem middlewares, loggers ou validadores por padrão, para simplicidade; adicione conforme necessidade.

## Componentes da Arquitetura
### Estado (State)
- Um dicionário mutável `{str: BaseModel}` que representa o contexto do agente (ex: `{'goal': str, 'plan': dict, 'results': list, 'achieved': bool}`).
- Inicializado com um goal e atualizado atomicamente via reducer.
- Benefícios: Fácil serialização (ex: para persistência) e validação via schemas (Pydantic).

### Ações (Actions)
- Funções que recebem o estado atual e retornam uma tuple opcional `(key: str, value: BaseModel)` para atualização.
- Ações podem envolver consultas a LLMs para gerar structured outputs, que são então interpretadas.
- Tipos:
  - **Ações Padrão**: Definidas no sistema (ver seção abaixo).
  - **Ações Customizadas (Tools)**: Funções externas registradas, como chamadas a APIs ou scripts.

### Reducer
- Função simples que aplica atualizações ao estado: `state[key] = value`.
- Garante imutabilidade ou atomicidade (ex: crie cópias do estado se necessário para thread-safety).

### Store
- Objeto central que segura o estado e expõe um método `dispatch(action_func)`.
- Ao despachar: Chama a ação, obtém o resultado e aplica o reducer.
- Opcional: Integra logging ou validação pós-dispatch.

### Dispatcher
- Parte integrada ao store; responsável por invocar ações de forma síncrona ou assíncrona.

### Opcionais
- **Logger**: Registra ações e mudanças de estado.
- **Validator**: Usa schemas (ex: Pydantic) para validar estados ou ações.
- **Serializer/Deserializer**: Para persistir o estado (ex: JSON).

## Integração com Structured Outputs
Para evitar function calls e suportar mais LLMs, o TAgent usa structured outputs em ações que requerem inteligência (ex: planning ou decision-making):
- **Processo**:
  1. Em uma ação (ex: Plan), construa um prompt com o estado atual e instrua o LLM a responder em formato estruturado (ex: JSON: `{'action': str, 'params': dict, 'reasoning': str}`).
  2. Envie ao LLM (qualquer modelo que gere texto).
  3. Parse a saída (valide com Pydantic; retry se inválida).
  4. Interprete: Use o campo `'action'` para decidir o que despachar (ex: se 'execute', chame uma tool).
  5. Atualize o estado com o resultado.
- **Exemplo de Schema Estruturado**:
  ```json
  {
    "action": "plan" | "execute" | "summarize" | "evaluate",
    "params": {"key": "value"},  // Parâmetros específicos
    "reasoning": "Explicação do porquê dessa ação"
  }
  ```
- **Vantagens**: Flexibilidade (funciona com LLMs como Llama sem tool support); controle granular via loop.
- **Implementação Dica**: Use bibliotecas como `instructor` (para Pydantic + LLM) ou prompts com few-shot examples para alta adesão.

## Fluxo de Ações
Todas as interações ocorrem via ações despachadas ao store. Ações retornam atualizações para o estado, que é um dicionário.

## Loop Principal
O "cérebro" do agente é um loop autônomo que consulta LLMs via structured outputs quando necessário:
- **Inicialização**: Defina o estado inicial com um goal.
- **Loop** (while not achieved):
  1. **Consulta LLM para Decisão**: Use structured output para obter a próxima ação sugerida (ex: prompt: "Baseado no estado: {state}. Sugira a próxima ação em JSON.").
  2. **Parse e Dispatch**: Interprete o output e despache a ação correspondente.
  3. **Atualize e Avalie**: Aplique reducer; verifique goal via avaliação.
- **Fluxo Exemplo**: Plan -> Execute -> (Execute | Plan | Summarize) -> Goal Evaluation -> (Plan | Summarize | End).
- Isso forma um ciclo ReAct-like: Raciocínio (via LLM structured), Ação (dispatch), Observação (estado atualizado).

## Ações Padrão do Sistema
- **Plan**: Gera um plano via LLM structured output (ex: lista de passos).
- **Execute**: Executa uma ação/ferramenta baseada em structured output (ex: chama API e atualiza 'results').
- **Summarize**: Resume o contexto em uma saída final.
- **Goal Evaluation**: Avalia se o goal foi alcançado (pode usar LLM para raciocínio complexo).

## Ferramentas como Ações
- Ferramentas são ações customizadas registradas no agente (ex: `register_tool('fetch_data', func)`).
- No loop, se o structured output indicar uma ferramenta, despache-a.
- Exemplo: Uma ferramenta 'search_web' chama uma API externa e retorna `( 'results', dados )`.

## Vantagens e Considerações
- **Vantagens**: Determinístico (Redux), autônomo, compatível com LLMs variados via structured outputs, extensível.
- **Considerações**: Gerencie custos de LLM (minimize chamadas); adicione timeouts para loops; teste prompts para alta precisão de structured outputs.
- **Extensões Futuras**: Integração com multi-agentes, persistência de estado ou UI para monitoramento.

Este documento serve como blueprint para implementação. Versão: 0.1 (Data: 12/07/2025).

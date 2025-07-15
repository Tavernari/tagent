# Proposta de Arquitetura Refinada: TAgent Community Workflow

## 1. Visão Geral

Este documento descreve uma arquitetura robusta e escalável para orquestrar múltiplos agentes de IA em **grupos de trabalho**, com execução sequencial entre grupos e paralela dentro deles. A solução é projetada como um recurso nativo do TAgent, priorizando **modularidade, resiliência e configuração declarativa** através de YAML. O objetivo é permitir a criação de fluxos de trabalho complexos, como o de enriquecimento de CNPJ, de forma eficiente e manutenível.

## 2. Arquitetura e Design

### 2.1. Conceito Central

O fluxo de trabalho é modelado como um Grafo Acíclico Dirigido (DAG) linear, onde cada nó é um grupo de agentes.

**Fluxo Lógico:**
```
            ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
Contexto -> │     Grupo 1      │  ──> │     Grupo 2      │  ──> │     Grupo 3      │  ──> Resultado
(Inicial)   │  (Exec. Paralela)│      │  (Exec. Paralela)│      │  (Exec. Sequencial)│      Final
            └──────────────────┘      └──────────────────┘      └──────────────────┘
```

**Fluxo de Dados Dentro de um Grupo:**
```
Contexto do Workflow ┐
                     ├─> [Agente A] ──> Resultado A ┐
                     ├─> [Agente B] ──> Resultado B ├─> [Sumarizador do Grupo] ─> Sumário ─┐
                     └─> [Agente C] ──> Resultado C ┘                                      ├─> Contexto Atualizado
                                                                                          ┘
```

*   **Execução Paralela (Padrão):** Agentes dentro de um grupo executam simultaneamente para máxima eficiência.
*   **Execução Sequencial (Opcional):** Agentes dentro de um grupo executam em ordem, útil para tarefas com dependência interna.
*   **Propagação de Contexto com Namespace:** O resultado de cada grupo é agregado ao contexto principal sob um namespace (o nome do grupo), prevenindo conflitos e garantindo a rastreabilidade dos dados.
*   **Lógica Condicional:** A execução de um grupo pode ser condicionada ao estado do contexto, permitindo a criação de fluxos de trabalho dinâmicos (branches).

### 2.2. Estrutura de Dados (Implementação com Pydantic)

Utilizaremos Pydantic para validação de dados, tipagem forte e serialização, o que torna a configuração mais segura e a depuração mais fácil.

```python
# Core structures
from typing import Dict, List, Any, Optional, Literal
import asyncio
from pydantic import BaseModel, Field

class GroupExecutionMode(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"

# Usamos um BaseModel para a configuração da ferramenta, permitindo validação.
# Ferramentas específicas (ex: WebSearchToolConfig) podem herdar desta classe.
class BaseToolConfig(BaseModel):
    tools: List[str]
    prompt: str
    model: str = "gpt-4-turbo"
    # Outros parâmetros comuns podem ser adicionados aqui

class AgentDefinition(BaseModel):
    name: str
    tool_config: BaseToolConfig
    timeout: int = Field(180, description="Timeout em segundos para a execução do agente.")
    required: bool = Field(True, description="Se a falha deste agente deve falhar o grupo (a menos que min_success_rate seja atingido).")

class WorkflowGroup(BaseModel):
    name: str
    agents: List[AgentDefinition]
    execution_mode: GroupExecutionMode = GroupExecutionMode.PARALLEL
    timeout: int = Field(300, description="Timeout total para o grupo.")
    min_success_rate: float = Field(1.0, description="Taxa mínima de sucesso de agentes 'required' para o grupo ser considerado bem-sucedido.")
    summary_prompt: Optional[str] = Field(None, description="Prompt para um LLM sumarizar os resultados deste grupo.")
    condition: Optional[str] = Field(None, description="Expressão Python a ser avaliada contra o contexto para executar este grupo. Ex: 'results.group_1.data.is_valid == True'")

class WorkflowDefinition(BaseModel):
    name: str
    groups: List[WorkflowGroup]
    initial_context: Dict[str, Any]
    final_summary_prompt: Optional[str] = Field(None, description="Prompt final para gerar um relatório abrangente a partir do contexto final.")

```

### 2.3. Fluxo de Execução do `WorkflowExecutor`

1.  **Inicialização:** Carrega a `WorkflowDefinition` e o `initial_context`.
2.  **Iteração de Grupos:** Percorre a lista `groups` sequencialmente.
3.  **Avaliação da Condição:** Para cada grupo, se o campo `condition` existir, avalia a expressão contra o contexto atual. Se for `False`, o grupo é pulado.
4.  **Execução do Grupo:**
    *   Cria tarefas `asyncio` para cada agente do grupo.
    *   Executa as tarefas (usando `asyncio.gather` para paralelo ou um loop `await` para sequencial).
    *   Gerencia os `timeouts` individuais e do grupo.
5.  **Coleta de Resultados:** Agrega os resultados de cada agente (sucesso ou falha) em um dicionário.
6.  **Validação de Sucesso:** Calcula a taxa de sucesso dos agentes `required`. Se for menor que `min_success_rate`, lança uma exceção e encerra o fluxo.
7.  **Sumarização (Opcional):** Se `summary_prompt` for definido, chama um LLM com o prompt e os resultados coletados para gerar um sumário conciso.
8.  **Atualização de Contexto:**
    *   Adiciona os resultados brutos dos agentes ao contexto principal sob `context['results'][group.name]`.
    *   Se um sumário foi gerado, adiciona-o em `context['summaries'][group.name]`.
    *   Esta abordagem com namespace previne sobreposição de dados.
9.  **Sumário Final:** Após a execução de todos os grupos, se `final_summary_prompt` existir, gera um relatório final usando todo o contexto acumulado.
10. **Retorno:** Retorna o contexto final completo, contendo todos os resultados, sumários e o relatório final.

## 3. Exemplo de Uso (YAML Refinado)

O YAML agora reflete a estrutura mais rica, incluindo `summary_prompt` e `condition`.

```yaml
name: "cnpj_enrichment_v2"
initial_context:
  cnpj: "{{ input.cnpj }}"
  razao_social: "{{ input.razao_social }}"

groups:
  - name: "first_line_research"
    execution_mode: "parallel"
    min_success_rate: 0.5 # Apenas 1 dos 2 agentes 'required' precisa funcionar
    summary_prompt: "Com base nas buscas no Reclame Aqui e Glassdoor, resuma o sentimento geral e os principais pontos de atenção sobre a empresa {{ razao_social }}."
    agents:
      - name: "reclame_aqui"
        required: true
        tool_config:
          tools: ["web_search"]
          prompt: "Busque por reclamações da empresa '{{ razao_social }}' (CNPJ {{ cnpj }}) no Reclame Aqui. Foque na reputação geral e em problemas recorrentes."
      - name: "glassdoor"
        required: true
        tool_config:
          tools: ["web_search"]
          prompt: "Busque por avaliações da empresa '{{ razao_social }}' no Glassdoor. Foque na cultura e satisfação dos funcionários."
      - name: "key_people_search"
        required: false # Agente opcional, sua falha não impacta o success_rate
        tool_config:
          tools: ["linkedin_search"]
          prompt: "Encontre os principais executivos (C-level) associados à empresa '{{ razao_social }}'."

  - name: "website_validation"
    execution_mode: "sequential"
    agents:
      - name: "official_website_finder"
        tool_config:
          tools: ["web_search"]
          prompt: "Encontre o website oficial da empresa '{{ razao_social }}'."
      - name: "website_scraper"
        tool_config:
          tools: ["web_scraper"]
          # O resultado do agente anterior pode ser injetado no prompt do próximo
          prompt: "Analise o conteúdo do site {{ results.website_validation.official_website_finder.url }} para validar se pertence à empresa e extraia informações de contato."

  - name: "social_media_presence"
    # Este grupo só executa se um site válido foi encontrado no grupo anterior
    condition: "results.website_validation.website_scraper.is_valid == true"
    execution_mode: "parallel"
    summary_prompt: "Resuma a presença e o engajamento da empresa nas redes sociais."
    agents:
      - name: "linkedin"
        tool_config:
          tools: ["linkedin_search"]
          prompt: "Busque o perfil oficial da empresa '{{ razao_social }}' no LinkedIn."

final_summary_prompt: "Gere um relatório de enriquecimento completo para o CNPJ {{ cnpj }}, consolidando os dados de reputação, website e presença social coletados."
```

## 4. Benefícios da Arquitetura Refinada

1.  **Robustez e Segurança:** O uso de Pydantic previne erros de configuração e garante a integridade dos dados em tempo de execução.
2.  **Tolerância a Falhas Inteligente:** `min_success_rate` e `required` oferecem controle granular sobre a resiliência do fluxo.
3.  **Fluxos Dinâmicos:** A lógica condicional (`condition`) permite criar "branches" e adaptar o workflow com base em resultados intermediários.
4.  **Contexto Organizado:** O uso de namespaces evita colisões de dados e torna o contexto final fácil de ser percorrido e depurado.
5.  **Manutenibilidade:** A configuração declarativa em YAML, separada da lógica de execução, facilita a criação e modificação de workflows por não-desenvolvedores.
6.  **Escalabilidade:** A arquitetura `async` nativa garante alta performance e utilização eficiente de recursos.

## 5. Roteiro de Implementação Sugerido

1.  **Fase 1 (Fundação):**
    *   Implementar as classes Pydantic (`AgentDefinition`, `WorkflowGroup`, etc.).
    *   Criar o `WorkflowExecutor` com a lógica principal de execução de grupos paralelos.
    *   Implementar a propagação de contexto com namespace.

2.  **Fase 2 (Configuração e Usabilidade):**
    *   Desenvolver o parser de YAML para `WorkflowDefinition`.
    *   Integrar um motor de templates (como Jinja2) para renderizar os contextos (`{{ ... }}`).

3.  **Fase 3 (Funcionalidades Avançadas):**
    *   Implementar a lógica de `summary_prompt` para sumarização por grupo.
    *   Implementar o avaliador de `condition` para execução condicional.
    *   Adicionar suporte a `execution_mode: sequential`.

4.  **Fase 4 (Resiliência e Operação):**
    *   Implementar mecanismos de `retry` com backoff exponencial para agentes.
    *   Adicionar logging estruturado detalhado para monitoramento e depuração.

5.  **Fase 5 (Otimização e Análise):**
    *   Criar um painel de visualização para acompanhar a execução dos workflows.
    *   Coletar métricas de performance (tempo por agente, taxa de falha, etc.) para otimização contínua.
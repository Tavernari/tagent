# TAgent RAG Context System - Development Tracking

## 📋 **Visão Geral**
Sistema de contexto RAG para TAgent que substitui instruções hardcoded por contexto dinâmico baseado em estado atual + embeddings/busca semântica.

## ✅ **Trabalho Realizado (Fase 1)**

### 1. **Estrutura de Memória Implementada**
- **Arquivo**: `src/tagent/models.py`
- **Funcionalidade**: 
  - `MemoryItem`: Modelo para armazenar fatos/conhecimento
  - `StructuredResponse`: Expandido com campo `memories`
  - Tipos: `'fact', 'pattern', 'strategy', 'lesson', 'context'`

### 2. **RAG Context Manager Básico**
- **Arquivo**: `src/tagent/rag_context.py`
- **Funcionalidade**:
  - `SimpleRagContextManager`: Armazenamento in-memory com busca por keywords
  - Contexto específico para PLAN, EXECUTE, EVALUATE
  - Inicialização com instruções básicas
  - Estatísticas de memória

### 3. **Integração com Agent Loop**
- **Arquivo**: `src/tagent/agent.py`
- **Funcionalidade**:
  - RAG context manager inicializado com goal
  - Processamento automático de memories das respostas LLM
  - Injeção de contexto nos prompts
  - Estatísticas no resultado final

### 4. **Exemplo de Uso**
- **Arquivo**: `examples/rag_memory_example.py`
- **Funcionalidade**: Demonstração completa do sistema RAG

## 🎯 **Nova Proposta (Fase 2) - Estado Atual da Análise**

### **Conceito Principal**
- **Eliminar instruções hardcoded**: Tudo vem do RAG
- **State Machine Linear**: `INIT → PLAN → EXECUTE → EVALUATE → [PASS→FINALIZE | FAIL→PLAN]`
- **Contexto Baseado em Estado**: `current_state + embeddings → prompt otimizado`
- **Histórico Contextual**: State carrega informação do "porquê" da volta para PLAN

### **Exemplo do Fluxo Desejado**
```
1. EVALUATE falha: "Faltou nome do usuário"
2. State recebe: last_action="evaluate", failure_reason="missing_user_name"
3. PLAN busca no RAG: contexto para "planning after evaluation failure about missing user name"
4. RAG retorna: instruções específicas sobre como coletar dados de usuário
5. PLAN cria tasks contextualizadas com o problema específico
```

## 🚧 **Próximos Passos Identificados**

### **Fase 2A: Expansão do State**
- **Objetivo**: State deve carregar contexto histórico para RAG
- **Implementação**:
  ```python
  state = {
      "goal": "...",
      "current_phase": "plan",
      "last_action": "evaluate", 
      "last_result": "failed",
      "failure_reason": "missing_user_name",
      "context_history": [...],
      "available_tools": [...],
      "collected_data": {...}
  }
  ```

### **Fase 2B: Context Retrieval Inteligente**
- **Objetivo**: Busca contextual baseada em state + embeddings
- **Implementação**:
  ```python
  def get_context_for_current_state(self, state: Dict) -> str:
      # Construir query baseada no estado atual
      query = f"{state['current_phase']} after {state['last_action']} {state['last_result']}"
      if state.get('failure_reason'):
          query += f" because {state['failure_reason']}"
      
      # Buscar contexto relevante
      return self._semantic_search(query, state)
  ```

### **Fase 2C: State Machine Simplificada**
- **Objetivo**: Fluxo linear com contexto histórico
- **Implementação**:
  ```python
  class LinearStateMachine:
      def __init__(self):
          self.states = [INIT, PLAN, EXECUTE, EVALUATE, FINALIZE]
          self.current_index = 0
      
      def transition(self, success: bool, context: Dict):
          if success:
              self.current_index += 1
          else:
              self.current_index = 1  # Volta para PLAN
              # Preserva contexto do porquê voltou
  ```

### **Fase 2D: Remoção de Instructions Hardcoded**
- **Objetivo**: Migrar todas as instruções para RAG
- **Implementação**:
  - Remover prompts hardcoded de `actions.py`
  - Carregar tudo dinamicamente do RAG
  - Inicializar RAG com instruções base via seed data

## 🛠️ **Estrutura Técnica Proposta**

### **Novo State Schema**
```python
class AgentState(BaseModel):
    goal: str
    current_phase: str
    last_action: Optional[str] = None
    last_result: Optional[str] = None  # "success" | "failed" | "partial"
    failure_reason: Optional[str] = None
    iteration: int = 0
    available_tools: List[str] = []
    collected_data: Dict[str, Any] = {}
    context_history: List[Dict] = []
```

### **Enhanced RAG Context Manager**
```python
class EnhancedRagContextManager:
    def get_context_for_current_state(self, state: AgentState) -> str:
        """Busca contexto baseado no estado atual completo"""
        
    def get_prompt_for_phase(self, phase: str, state: AgentState) -> str:
        """Gera prompt completo para a fase atual"""
        
    def store_execution_context(self, state: AgentState, result: Any):
        """Armazena contexto da execução para futuras referências"""
```

### **Simplified State Machine**
```python
class LinearAgentStateMachine:
    FLOW = [INIT, PLAN, EXECUTE, EVALUATE, FINALIZE]
    
    def get_next_action(self, current_state: AgentState) -> ActionType:
        """Retorna próxima ação baseada no estado atual"""
        
    def handle_evaluation_result(self, success: bool, reason: str) -> AgentState:
        """Maneja resultado da avaliação e atualiza estado"""
```

## 🎯 **Benefícios Esperados**

### **Técnicos**
- **Simplicidade**: Fluxo linear mais fácil de entender e debugar
- **Flexibilidade**: Contexto dinâmico se adapta a qualquer scenario
- **Escalabilidade**: RAG pode crescer indefinidamente
- **Manutenibilidade**: Sem instruções hardcoded para manter

### **Funcionais**
- **Contexto Inteligente**: Sempre sabe "porquê" está fazendo algo
- **Aprendizado Contínuo**: Cada execução melhora o contexto
- **Recuperação Inteligente**: Falhas geram contexto para correção
- **Adaptabilidade**: Sistema se adapta a diferentes domínios

## 🔄 **Estado Atual do Desenvolvimento**

### **Completado** ✅
- [x] Estrutura básica de memória
- [x] RAG Context Manager simples
- [x] Integração com agent loop
- [x] Exemplo funcional

### **Em Análise** 🔍
- [x] Proposta de state machine linear
- [x] Definição de estrutura de state expandido
- [x] Planejamento de context retrieval inteligente

### **Próximo** 🚧
- [ ] Implementar state expandido
- [ ] Criar enhanced RAG context manager
- [ ] Simplificar state machine
- [ ] Migrar instruções para RAG
- [ ] Testes com cenários reais

## 📝 **Decisões Pendentes**

1. **Embeddings**: Usar qual modelo? OpenAI, Sentence-Transformers, local?
2. **Persistência**: Manter RAG em memória ou adicionar persistência?
3. **Backward Compatibility**: Manter compatibilidade com sistema atual?
4. **Performance**: Como otimizar busca em RAG grandes?
5. **Fallbacks**: O que fazer se RAG não encontrar contexto relevante?

## 🧪 **Testes Planejados**

1. **Cenário de Falha**: Evaluator rejeita por dados específicos faltando
2. **Cenário de Loop**: Múltiplas tentativas de PLAN→EXECUTE→EVALUATE
3. **Cenário de Sucesso**: Fluxo completo sem falhas
4. **Cenário de Recuperação**: Falha → contexto → correção → sucesso

---

**Última Atualização**: 2025-01-15  
**Próxima Revisão**: Após implementação da Fase 2A  
**Responsável**: Victor Tavernari + Claude
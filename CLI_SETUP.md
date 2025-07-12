# TAgent CLI Setup Guide

## ✅ Problema Resolvido!

O erro que você encontrou:
```
Traceback (most recent call last):
  File "/Users/victortavernari/Library/Python/3.9/bin/tagent", line 5, in <module>
    from main import main
ModuleNotFoundError: No module named 'main'
```

Foi causado pela instalação no Python do sistema em vez do virtual environment. Agora está corrigido!

## 🚀 Instalação Correta

### Opção 1: Instalação Limpa (Recomendada)
```bash
make clean-install
source .venv/bin/activate
tagent --help
```

### Opção 2: Instalação Manual
```bash
# Criar venv se não existir
python3 -m venv .venv

# Ativar o ambiente virtual
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Testar
tagent --help
```

## 🔧 Comandos Disponíveis

### Comandos Make
```bash
make help                  # Ver todos os comandos
make clean-install         # Instalação limpa
make cli-help             # Mostrar ajuda do CLI
make cli-discovery-test   # Testar descoberta de ferramentas
make cli-test             # Teste completo com exemplo de viagem
make cli-demo             # Gerar GIF de demonstração
```

### Comandos CLI Diretos (após `source .venv/bin/activate`)
```bash
# Comando principal
tagent "Plan a trip to Rome" --search-dir examples/travel_planning_cli

# Ou usando python -m
python -m tagent "Plan a trip to Rome" --search-dir examples/travel_planning_cli

# Ver ajuda
tagent --help

# Teste rápido de descoberta
tagent "Test" --search-dir examples/travel_planning_cli --max-iterations 1
```

## 📁 Estrutura de Arquivos

```
tagent2/
├── src/tagent/
│   ├── __init__.py
│   ├── __main__.py      # Permite python -m tagent
│   ├── cli.py           # CLI principal
│   └── agent.py         # Core do TAgent
├── examples/
│   └── travel_planning_cli/
│       ├── tagent.tools.py    # 5 ferramentas de viagem
│       └── tagent.output.py   # Schema TravelPlan
├── Makefile             # Comandos de build
└── setup.py             # Configuração do console script
```

## 🎯 Como Usar

### 1. Criar suas próprias ferramentas
Crie um arquivo `meu_projeto/tagent.tools.py`:
```python
def minha_ferramenta(state, args):
    # Sua lógica aqui
    return ('resultado', 'valor')
```

### 2. Criar esquema de saída
Crie um arquivo `meu_projeto/tagent.output.py`:
```python
from pydantic import BaseModel

class MeuResultado(BaseModel):
    campo1: str
    campo2: int

output_schema = MeuResultado
```

### 3. Executar
```bash
tagent "Meu objetivo" --search-dir meu_projeto
```

## ⚠️ Pontos Importantes

1. **Sempre use o virtual environment**: `source .venv/bin/activate`
2. **Use `make clean-install`** se tiver problemas de instalação
3. **Teste com**: `make cli-discovery-test`
4. **Console script**: `tagent` funciona diretamente após instalação
5. **Module mode**: `python -m tagent` sempre funciona

## 🎉 Benefícios do CLI

- ✅ Descoberta automática de ferramentas
- ✅ Modularidade e reutilização
- ✅ Console script instalável
- ✅ Ambiente virtual isolado
- ✅ Comandos Make convenientes
- ✅ Exemplos funcionais inclusos

Agora o TAgent CLI está pronto para produção! 🚀
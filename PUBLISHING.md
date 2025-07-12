# TAgent - Publishing Guide

Este guia fornece instruções completas para publicar o TAgent no PyPI.

## 📋 Pré-requisitos

1. **Contas PyPI**:
   - Criar conta no [TestPyPI](https://test.pypi.org) para testes
   - Criar conta no [PyPI](https://pypi.org) para produção

2. **Dependências de Build**:
   ```bash
   pip install build twine
   ```

3. **Configuração de API Tokens** (Recomendado):
   - Gere tokens de API no PyPI/TestPyPI
   - Configure no `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-AgEIcHlwaS5vcmcC...

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-AgEIcHlwaS5vcmcC...
   ```

## 🚀 Processo de Publicação Automatizado

### Script de Publicação

O projeto inclui um script automatizado em `scripts/publish.py`:

```bash
# Teste completo no TestPyPI
python scripts/publish.py --target test

# Publicação em produção no PyPI
python scripts/publish.py --target prod

# Apenas limpeza dos artefatos
python scripts/publish.py --clean-only

# Pular testes (não recomendado)
python scripts/publish.py --target test --skip-tests
```

### O que o Script Faz

1. **Limpeza**: Remove artefatos de build anteriores
2. **Testes**: Executa a suíte de testes
3. **Linting**: Verifica qualidade do código
4. **Build**: Cria distribuições wheel e sdist
5. **Validação**: Verifica se as distribuições são válidas
6. **Upload**: Envia para PyPI/TestPyPI

## 🔧 Processo Manual

### 1. Preparação

```bash
# Limpar artefatos anteriores
rm -rf build/ dist/ src/tagent.egg-info/

# Verificar qualidade do código
make lint
make test
```

### 2. Build das Distribuições

```bash
# Instalar dependências de build
pip install build twine

# Criar distribuições
python -m build

# Verificar distribuições
python -m twine check dist/*
```

### 3. Teste no TestPyPI

```bash
# Upload para TestPyPI
python -m twine upload --repository testpypi dist/*

# Testar instalação
pip install --index-url https://test.pypi.org/simple/ tagent
```

### 4. Publicação Final

```bash
# Upload para PyPI
python -m twine upload dist/*
```

## 📦 Estrutura do Pacote

O projeto está configurado com:

```
tagent/
├── pyproject.toml          # Configuração moderna do projeto
├── MANIFEST.in             # Arquivos incluídos na distribuição
├── src/tagent/             # Código fonte
│   ├── __init__.py
│   ├── agent.py
│   ├── cli.py
│   └── version.py
├── examples/               # Exemplos incluídos
├── tests/                  # Suíte de testes
└── scripts/publish.py      # Script de publicação
```

## 🔄 Versionamento

Para nova versão:

1. **Atualizar versão** em `pyproject.toml`:
   ```toml
   [project]
   version = "0.2.2"  # Siga semantic versioning
   ```

2. **Commit e tag**:
   ```bash
   git commit -m "Bump version to 0.2.2"
   git tag v0.2.2
   git push origin main --tags
   ```

3. **Publicar nova versão**:
   ```bash
   python scripts/publish.py --target prod
   ```

## ✅ Checklist de Publicação

Antes de publicar, verifique:

- [ ] Testes passando (`make test`)
- [ ] Linting limpo (`make lint`)
- [ ] Versão atualizada no `pyproject.toml`
- [ ] README.md atualizado
- [ ] CHANGELOG.md atualizado (se existir)
- [ ] Exemplos funcionando
- [ ] Testado no TestPyPI

## 🛠️ Comandos Úteis

```bash
# Verificar package info
python -m pip show tagent

# Listar arquivos incluídos
python -m build --sdist && tar -tzf dist/tagent-*.tar.gz

# Verificar dependências
python -m pip check

# Instalar versão de desenvolvimento
pip install -e .
```

## 🔍 Troubleshooting

### Erro de Upload
```
HTTP Error 403: The user ... isn't allowed to upload to project 'tagent'
```
**Solução**: Verifique se você é o proprietário do projeto ou use um nome diferente.

### Dependências Faltando
```
ModuleNotFoundError: No module named 'pydantic'
```
**Solução**: Verifique as dependências no `pyproject.toml` e `requirements.txt`.

### Problemas de Formatação
```
error: Microsoft Visual C++ 14.0 is required
```
**Solução**: Use distribuições wheel (`--universal`) ou especifique melhor as dependências.

## 📚 Recursos Adicionais

- [PyPI Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)
- [Semantic Versioning](https://semver.org/)

---

**Última atualização**: 12 de julho de 2025
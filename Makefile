.PHONY: help install test lint format clean clean-install cli-help cli-test cli-discovery-test cli-demo build

help:
	@echo "Available commands:"
	@echo "  make install            Install dependencies and CLI in .venv"
	@echo "  make clean-install      Clean previous installs and reinstall"
	@echo "  make test              Run tests"
	@echo "  make lint              Run linters"
	@echo "  make format            Format code"
	@echo "  make clean             Clean up build files"
	@echo ""
	@echo "CLI Commands (require 'source .venv/bin/activate'):"
	@echo "  make cli-help          Show CLI help"
	@echo "  make cli-discovery-test Test CLI tool discovery"
	@echo "  make cli-test          Test CLI with travel example"
	@echo "  make cli-demo          Generate CLI demo GIF"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build             Build distribution packages"
	@echo ""
	@echo "Quick Start:"
	@echo "  make clean-install     # Clean install everything"
	@echo "  source .venv/bin/activate  # Activate virtual environment"
	@echo "  tagent --help          # Use CLI directly"

install:
	@echo "Installing TAgent in virtual environment..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv .venv; \
	fi
	@echo "Activating virtual environment and installing dependencies..."
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -r requirements-dev.txt
	. .venv/bin/activate && pip install -e .
	@echo "✅ Installation complete! Activate with: source .venv/bin/activate"

test:
	pytest

lint:
	flake8 src tests
	mypy src tests

format:
	isort .
	black .

# CLI Commands
cli-help:
	@echo "TAgent CLI Help:"
	@. .venv/bin/activate && tagent --help

cli-test:
	@echo "Testing CLI with travel planning example..."
	@. .venv/bin/activate && tagent "Plan a quick trip from London to Rome for 2025-09-10 to 2025-09-17 with budget 1500 USD" \
		--search-dir examples/travel_planning_cli \
		--max-iterations 5 \
		--model openrouter/google/gemini-2.5-pro

cli-discovery-test:
	@echo "Testing CLI tool discovery..."
	@. .venv/bin/activate && tagent "Test discovery" --search-dir examples/travel_planning_cli --max-iterations 1 --verbose

cli-demo:
	@echo "Generating CLI demo GIF..."
	@if command -v vhs >/dev/null 2>&1; then \
		. .venv/bin/activate && vhs examples/tagent_cli_demo.tape; \
		echo "Demo GIF generated: examples/tagent_cli_demo.gif"; \
	else \
		echo "VHS not found. Install with: go install github.com/charmbracelet/vhs@latest"; \
		echo "Or use Homebrew: brew install vhs"; \
	fi

# Build Commands
build:
	@echo "Building distribution packages..."
	python -m build
	@echo "Build complete! Check dist/ directory"

# Development Commands
dev-install:
	@echo "Installing in development mode..."
	pip install -e .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete
	find . -type f -name '*.py[co]' -delete

clean-install:
	@echo "Cleaning previous installations and reinstalling..."
	@# Remove system-wide installation if it exists
	@if [ -f "/Users/victortavernari/Library/Python/3.9/bin/tagent" ]; then \
		echo "Removing system-wide tagent installation..."; \
		rm -f /Users/victortavernari/Library/Python/3.9/bin/tagent; \
	fi
	@# Uninstall from user site-packages if exists
	@python3 -m pip uninstall -y tagent 2>/dev/null || true
	@# Clean and reinstall in venv
	$(MAKE) clean
	$(MAKE) install
	@echo "✅ Clean installation complete!"

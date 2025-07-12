.PHONY: help install test lint format clean

help:
	@echo "Available commands:"
	@echo "  make install     Install dependencies"
	@echo "  make test       Run tests"
	@echo "  make lint       Run linters"
	@echo "  make format     Format code"
	@echo "  make clean      Clean up build files"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest

lint:
	flake8 src tests
	mypy src tests

format:
	isort .
	black .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete
	find . -type f -name '*.py[co]' -delete

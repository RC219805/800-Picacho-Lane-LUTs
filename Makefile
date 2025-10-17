.PHONY: help install install-dev lint format test security clean check-all

help:
	@echo "Available commands:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make lint          Run all linters"
	@echo "  make format        Format code with black and isort"
	@echo "  make test          Run tests with coverage"
	@echo "  make security      Run security checks"
	@echo "  make clean         Clean up generated files"
	@echo "  make check-all     Run all checks (lint, test, security)"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

format:
	@echo "Formatting with black..."
	black .
	@echo "Sorting imports with isort..."
	isort .

lint:
	@echo "Running flake8..."
	flake8 .
	@echo "Running pylint..."
	find . -name "*.py" -not -path "./01_Film_Emulation/*" -not -path "./02_Location_Aesthetic/*" -not -path "./03_Material_Response/*" -not -path "./venv/*" | xargs pylint --exit-zero

test:
	pytest -v --cov=. --cov-report=term-missing --cov-report=html || true

security:
	@echo "Running bandit security scan..."
	bandit -r . -x tests/,01_Film_Emulation/,02_Location_Aesthetic/,03_Material_Response/,venv/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/


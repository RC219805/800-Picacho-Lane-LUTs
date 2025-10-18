# path: Makefile
.PHONY: help install install-dev lint format test test-fast test-novideo test-full security clean check-all venv ci

# Defaults (overridable): use `make PY=python3.11` if needed
PY ?= python3
PIP ?= $(PY) -m pip
FAST_TESTS ?= -k "not video_master_grader" tests

help:
	@echo "Targets:"
	@echo "  venv          Create local .venv if missing"
	@echo "  install       Install project (and requirements.txt if present)"
	@echo "  install-dev   Install dev tools (requirements-dev.txt, pre-commit, linters)"
	@echo "  format        Run isort then black (line-length 127)"
	@echo "  lint          flake8 critical + pylint (non-blocking)"
	@echo "  test          Run pytest on ./tests"
	@echo "  test-fast     Run fast subset (default: $(FAST_TESTS))"
	@echo "  test-novideo  Run all tests excluding video suite"
	@echo "  test-full     Run entire test suite (xdist auto if available)"
	@echo "  security      Run bandit (skip tests/, B101,B601)"
	@echo "  clean         Remove build artifacts, caches, coverage"
	@echo "  check-all     Run pre-commit (if installed) then test-novideo"
	@echo "  ci            Run lint + test-fast"

venv:
	@if [ ! -x .venv/bin/python ]; then \
		"$(PY)" -m venv .venv && echo "Created .venv"; \
	else \
		echo ".venv already present"; \
	fi

install:
	@$(PIP) install --upgrade pip
	@if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; fi
	@$(PIP) install .

install-dev: install
	@if [ -f requirements-dev.txt ]; then $(PIP) install -r requirements-dev.txt; fi
	@$(PIP) install pre-commit black isort flake8 pylint bandit mypy || true
	@command -v pre-commit >/dev/null 2>&1 && pre-commit install || true

format:
	@$(PY) -m isort . --profile black --line-length 127
	@$(PY) -m black . --line-length 127

lint:
	@echo "Running flake8 critical checks..."
	@$(PY) -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
	@echo "Running pylint (non-blocking)..."
	@files=$$(git ls-files '*.py' || true); \
	if [ -n "$$files" ]; then $(PY) -m pylint $$files || true; else echo "No Python files."; fi

test:
	@"$(PY)" -m pytest -q tests

test-fast:
	@"$(PY)" -m pytest -q $(FAST_TESTS)

test-novideo:
	@"$(PY)" -m pytest -q -k 'not video_master_grader'

test-full:
	@if "$(PY)" -m pip list | grep -q pytest-xdist; then \
		"$(PY)" -m pytest -q -n auto tests; \
	else \
		"$(PY)" -m pytest -q tests; \
	fi

security:
	@$(PY) -m bandit -q -r . -x tests/ -s B101,B601 || true

clean:
	@echo "Cleaning build artifacts, caches, coverage..."
	@rm -rf build dist .eggs *.egg-info
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	@rm -rf .pytest_cache .mypy_cache .ruff_cache || true
	@rm -rf htmlcov .coverage* coverage.xml || true

check-all:
	@echo "Running pre-commit (if installed)..."
	@command -v pre-commit >/dev/null 2>&1 && pre-commit run --all-files --show-diff-on-failure || echo "pre-commit not installed"
	@$(MAKE) test-novideo

ci: lint test-fast
	@echo "✅ Local CI checks completed successfully."
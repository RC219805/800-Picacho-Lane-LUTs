SHELL := /bin/sh

# Resolve a Python interpreter: prefer local venv, otherwise fall back to python3
PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python3 || command -v python; fi)

# Common subsets (fast tests avoid heavy/optional paths)
FAST_TESTS := \
	tests/test_material_response.py \
	tests/test_adjustments.py \
	tests/test_geometry.py \
	tests/test_io.py \
	tests/test_pipeline.py \
	tests/test_presets.py

.PHONY: help test-fast test-novideo test-full venv

help:
	@echo "Targets:"
	@echo "  test-fast     Run fast subset (no video/optional heavy paths)"
	@echo "  test-novideo  Run all tests excluding video suite via -k filter"
	@echo "  test-full     Run entire test suite (parallel if xdist present)"
	@echo "  venv          Create local .venv if missing"

venv:
	@if [ ! -x .venv/bin/python ]; then \
		"$(PY)" -m venv .venv && echo "Created .venv" || true; \
	else \
		echo ".venv already present"; \
	fi

test-fast:
	@"$(PY)" -m pytest -q $(FAST_TESTS)

test-novideo:
	@"$(PY)" -m pytest -q -k 'not video_master_grader'

test-full:
	@if "$(PY)" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('xdist') else 1)"; then \
		"$(PY)" -m pytest -q -n auto tests; \
	else \
		"$(PY)" -m pytest -q tests; \
	fi

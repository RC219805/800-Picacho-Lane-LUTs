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

.PHONY: help test-fast test-novideo test-full venv pipeline-run pipeline-tensor pipeline-optimize pipeline-render

help:
	@echo "Targets:"
	@echo "  test-fast     Run fast subset (no video/optional heavy paths)"
	@echo "  test-novideo  Run all tests excluding video suite via -k filter"
	@echo "  test-full     Run entire test suite (parallel if xdist present)"
	@echo "  venv          Create local .venv if missing"
	@echo "  pipeline-run      Execute the material intelligence pipeline"
	@echo "  pipeline-tensor   Build tensors only"
	@echo "  pipeline-optimize Optimize assignments only"
	@echo "  pipeline-render   Render lighting scenarios"

venv:
	@if [ ! -x .venv/bin/python ]; then \
		"$(PY)" -m venv .venv && echo "Created .venv"; \
	else \
		echo ".venv already present"; \
	fi

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

pipeline-run:
	"$(PY)" pipeline_cli.py run $(RUN_ARGS)

pipeline-tensor:
	"$(PY)" pipeline_cli.py tensor --id-mask $(ID) --output-dir $(OUT) $(if $(PIXEL_SIZE),--pixel-size $(PIXEL_SIZE),)

pipeline-optimize:
	"$(PY)" pipeline_cli.py optimize --baseline-id-mask $(ID) --baseline-palette $(PAL) --output-dir $(OUT) $(if $(PIXEL_SIZE),--pixel-size $(PIXEL_SIZE),) $(if $(GEN),--generations $(GEN),) $(if $(POP),--population $(POP),)

pipeline-render:
	"$(PY)" pipeline_cli.py render --id-mask $(ID) --output-dir $(OUT) $(if $(SCNS),--scenarios $(SCNS),) $(if $(PIXEL_SIZE),--pixel-size $(PIXEL_SIZE),)

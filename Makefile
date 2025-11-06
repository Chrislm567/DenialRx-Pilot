SHELL := /bin/bash
PYTHON ?= python3
VENV_BIN := .venv/bin

.PHONY: bootstrap lint format format-check type-check test scan pii-scan clean

bootstrap:
	pnpm install
	@if [ ! -d .venv ]; then \
		$(PYTHON) -m venv .venv; \
	fi
	@. $(VENV_BIN)/activate && pip install --upgrade pip && pip install -r requirements-dev.txt

lint:
	pnpm lint
	@. $(VENV_BIN)/activate && ruff check packages tests tools

format:
	pnpm format
	@. $(VENV_BIN)/activate && ruff check --fix packages tests tools

format-check:
	pnpm run format:check
	@. $(VENV_BIN)/activate && ruff check packages tests tools

type-check:
	@. $(VENV_BIN)/activate && mypy packages/rules/src

test:
	$(PYTHON) -m unittest discover -s tests

scan:
	@. $(VENV_BIN)/activate && bandit -r packages -ll

pii-scan:
	$(PYTHON) tools/pii_scan.py

clean:
	rm -rf node_modules .venv .mypy_cache .ruff_cache __pycache__

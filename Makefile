.PHONY: help run clean venv install test test-cov lint

SERVICE_NAME  ?= "processor-post-timeseries"
VENV_DIR      ?= venv
PYTHON        ?= python3

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make venv     - create virtual environment and install all dependencies"
	@echo "make install  - install dependencies into existing venv"
	@echo "make test     - run tests"
	@echo "make test-cov - run tests with coverage report"
	@echo "make lint     - run linter"
	@echo "make run      - run the processor locally via docker-compose"
	@echo "make clean    - remove all files from locally mounted input / output directories"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r processor/requirements.txt
	$(VENV_DIR)/bin/pip install -r requirements-test.txt
	@echo ""
	@echo "Virtual environment created. Activate with:"
	@echo "  source $(VENV_DIR)/bin/activate"

install:
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r processor/requirements.txt
	$(VENV_DIR)/bin/pip install -r requirements-test.txt

test:
	$(VENV_DIR)/bin/python -m pytest tests/ -v

test-cov:
	$(VENV_DIR)/bin/python -m pytest tests/ -v --cov=processor --cov-report=term-missing

lint:
	$(VENV_DIR)/bin/pip install ruff --quiet
	$(VENV_DIR)/bin/ruff check processor/ tests/

run:
	docker-compose -f docker-compose.yml down --remove-orphans
	docker-compose -f docker-compose.yml build
	docker-compose -f docker-compose.yml up --exit-code-from processor

clean:
	rm -f data/input/*
	rm -f data/output/*

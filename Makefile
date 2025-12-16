.PHONY: help run clean test test-coverage

SERVICE_NAME  ?= "processor-post-timeseries"

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make run           - run the processor locally via docker-compose"
	@echo "make clean         - remove all files from locally mounted input / output directories"
	@echo "make test          - run tests"
	@echo "make test-coverage - run tests with code coverage reporting"

run:
	docker-compose -f docker-compose.yml down --remove-orphans
	docker-compose -f docker-compose.yml build
	docker-compose -f docker-compose.yml up --exit-code-from processor

clean:
	rm -f data/input/*
	rm -f data/output/*

test:
	source venv/bin/activate && python -m pytest tests/ -v

test-coverage:
	source venv/bin/activate && python -m pytest tests/ --cov=processor --cov-report=term-missing

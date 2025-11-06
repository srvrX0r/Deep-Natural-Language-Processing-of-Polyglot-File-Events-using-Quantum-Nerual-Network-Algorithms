# Makefile for Quantum Malware Detector

.PHONY: help build install clean test benchmark docs

help:
	@echo "Quantum Malware Detector - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  build      - Build Cython extensions"
	@echo "  install    - Install the package"
	@echo "  clean      - Clean build artifacts"
	@echo "  test       - Run test suite"
	@echo "  benchmark  - Run performance benchmarks"
	@echo "  docs       - Generate documentation"
	@echo "  deploy     - Deploy to production"

build:
	@echo "Building Cython extensions..."
	bash scripts/build.sh

install: build
	@echo "Installing package..."
	pip install -e .

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	find . -type f -name "*.so" -delete
	find . -type f -name "*.c" -path "*/src/*" -delete
	find . -type f -name "*.cpp" -path "*/src/*" -delete
	find . -type f -name "*.html" -path "*/src/*" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test: build
	@echo "Running test suite..."
	pytest tests/ -v

test-coverage: build
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term

benchmark: build
	@echo "Running performance benchmarks..."
	python scripts/benchmark.py

docs:
	@echo "Generating documentation..."
	@echo "Documentation available in README.md and USAGE.md"

deploy:
	@echo "Deploying to production..."
	sudo bash scripts/deploy_production.sh

lint:
	@echo "Running linters..."
	flake8 src/ --max-line-length=100 --ignore=E501,W503
	pylint src/ --disable=C0111,C0103

format:
	@echo "Formatting code..."
	black src/ tests/ scripts/*.py

check: lint test

all: clean build test

.DEFAULT_GOAL := help

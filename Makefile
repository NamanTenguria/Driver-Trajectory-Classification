# Makefile for ADM Assignment 3: Trajectory Analysis

.PHONY: help install test clean train predict setup

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up the project environment
	python -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	python -m pytest tests/ -v

train: ## Train the models
	python -m src.main

predict: ## Make predictions (requires trained models)
	python -c "from src.main import evaluate_models; evaluate_models('data/test.pkl')"

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

format: ## Format code with black
	black src/ tests/

lint: ## Run linting
	flake8 src/ tests/

type-check: ## Run type checking
	mypy src/

all-checks: format lint type-check test ## Run all code quality checks

create-dirs: ## Create necessary directories
	mkdir -p data models logs

init: create-dirs install ## Initialize the project

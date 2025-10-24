 # =====================================================================
# Wave Theory Chatbot - Makefile for Development and Deployment
# =====================================================================

.PHONY: help install dev test build run stop clean docker-build docker-run docker-stop

# Default target
help:
	@echo "Wave Theory Chatbot - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies"
	@echo "  dev         Run in development mode"
	@echo "  test        Run tests"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run with Docker Compose"
	@echo "  docker-stop     Stop Docker containers"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       Clean temporary files"
	@echo "  format      Format code with black"
	@echo "  lint        Run linting checks"
	@echo "  export-html Export standalone HTML version"
	@echo "  run-enhanced Run enhanced Streamlit app"

# Development commands
install:
	pip install -r requirements.txt

dev:
	python run_app.py

test:
	python test_integration.py

test-pytest:
	pytest tests/ -v

format:
	black src/ tests/
	flake8 src/ tests/

lint:
	flake8 src/ tests/

# Docker commands
docker-build:
	docker build -t wave-theory-chatbot .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

# Export commands
export-html:
	python export_html.py
	@echo "Standalone HTML exported to: wave_theory_standalone.html"

run-enhanced:
	streamlit run src/app/enhanced_streamlit_app.py --server.port 8501

# Full deployment
deploy: docker-build docker-run
	@echo "Wave Theory Chatbot deployed successfully!"
	@echo "Access at: http://localhost:8501"

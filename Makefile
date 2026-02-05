.PHONY: build run test clean docker-build docker-buildx docker-buildx-setup docker-run

# Python virtual environment
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Docker
IMAGE_NAME := 767397794709.dkr.ecr.us-west-2.amazonaws.com/hashicups/rag-api
IMAGE_TAG := v0.0.6

# Development
install: $(VENV)
	$(PIP) install -r requirements.txt

$(VENV):
	python3 -m venv $(VENV)

run: install
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m flake8 app/
	$(PYTHON) -m mypy app/

format:
	$(PYTHON) -m black app/
	$(PYTHON) -m isort app/

# Docker
docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Create buildx builder (run once if you get "no builder" errors)
docker-buildx-setup:
	docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch

# Multi-arch build (amd64 + arm64) and push via buildx
docker-buildx: docker-buildx-setup
	docker buildx build --platform linux/amd64,linux/arm64 \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		--provenance=false \
		--push .

docker-run:
	docker run -p 8080:8080 -p 9102:9102 \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e PUBLIC_API_URI=http://host.docker.internal:8080/api \
		$(IMAGE_NAME):$(IMAGE_TAG)

# Cleanup
clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Health check
health:
	curl -s http://localhost:8080/health/livez | jq .

stats:
	curl -s http://localhost:8080/health/stats | jq .

# Test chat endpoint
chat:
	curl -s -X POST http://localhost:8080/chat \
		-H "Content-Type: application/json" \
		-d '{"message": "What coffee has chocolate flavor?"}' | jq .

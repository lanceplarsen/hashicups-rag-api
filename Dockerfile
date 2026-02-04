# Multi-stage Dockerfile for HashiCups RAG API

# Stage 1: OpenTelemetry Collector
FROM otel/opentelemetry-collector:0.91.0 AS otel-collector

# Stage 2: Python application
FROM python:3.11-slim

# Install OpenTelemetry Collector from stage 1
COPY --from=otel-collector /otelcol /usr/local/bin/otelcol

# Create non-root user with home directory
RUN groupadd -r api && useradd -r -g api -m -d /home/api api

# Set working directory
WORKDIR /app

# Create data directory for Chroma persistence and model cache
RUN mkdir -p /app/data/chroma /app/cache && chown -R api:api /app/data /app/cache

# Set cache directories for HuggingFace models
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY conf.json .
COPY otel-collector-config.yaml .
COPY docker-entrypoint.sh /usr/local/bin/

# Make entrypoint executable
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Change ownership
RUN chown -R api:api /app

# Switch to non-root user
USER api

# Expose ports
EXPOSE 8080 9102

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

#!/bin/bash
set -e

# Signal handling for graceful shutdown
cleanup() {
    echo "Shutting down services..."
    kill -TERM "$OTEL_PID" 2>/dev/null || true
    kill -TERM "$APP_PID" 2>/dev/null || true
    wait "$OTEL_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start OpenTelemetry Collector
echo "Starting OpenTelemetry Collector..."
/usr/local/bin/otelcol --config=/app/otel-collector-config.yaml &
OTEL_PID=$!

# Wait for OTel Collector to start
sleep 2

# Start application
echo "Starting RAG API..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 &
APP_PID=$!

# Wait for either process to exit
wait -n

# If we get here, one process exited - clean up
cleanup

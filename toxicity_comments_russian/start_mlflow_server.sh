#!/bin/bash

uv run mlflow server \
    --backend-store-uri ./plots \
    --default-artifact-root ./plots \
    --host 127.0.0.1 \
    --port 8080

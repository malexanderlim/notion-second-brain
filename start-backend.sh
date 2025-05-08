#!/bin/bash
# Activate venv and start the FastAPI server from the project root

echo "Starting Backend Server (uvicorn)..."

# Activate virtual environment (adjust path if necessary)
source ./venv/bin/activate

# Start FastAPI server from project root, pointing to the app object
# Add --host 0.0.0.0 if you need to access it from other devices on your network
uvicorn backend.main:app --reload --app-dir . --host 0.0.0.0 
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY drag_llm_service/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy drag_python_client library
COPY drag_python_client /app/drag_python_client

# Copy drag_contract for ABI access
COPY drag_contract /app/drag_contract

# Copy application code
COPY drag_llm_service /app

# Expose port
EXPOSE 9000

# Run the server
CMD ["python", "app/server.py"]

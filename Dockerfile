FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install CPU-only PyTorch first (much smaller than CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sentence-transformers \
    numpy \
    click \
    tqdm \
    scikit-learn \
    requests

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /data

# Environment variables
ENV MOLTMIRROR_DB_PATH=/data/analysis.db
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

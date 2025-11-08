# Docker base image. Needs to be cuda to allow GPU usage
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY Event.py /app/
COPY Agent.py /app/
COPY EventAgent.py /app/
COPY EventType.py /app/
COPY AgentHandler.py /app/
COPY Model.py /app/
COPY Moderation.py /app/
COPY ModerationAgent.py /app/
COPY Summarizer.py /app/
COPY SummarizerAgent.py /app/

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Expose port for FastAPI
EXPOSE 8000

# Health check
#HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#    CMD curl -f http://localhost:8000/docs || exit 1

# Command to run the FastAPI application
CMD ["python", "-m", "uvicorn", "AgentHandler:app", "--host", "0.0.0.0", "--port", "8000"]

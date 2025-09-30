# =====================================================================
# Wave Theory Chatbot - Docker Configuration
# =====================================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and models
RUN mkdir -p data models checkpoints logs

# Set proper permissions
RUN chmod -R 755 /app

# Expose Streamlit port for local runs (Cloud Run injects $PORT)
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORM_NAME=cpu
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app (Cloud Run sets $PORT)
ENV PORT=8501
CMD ["bash", "-lc", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]

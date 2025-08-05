# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create .env file template (you'll need to provide actual values)
RUN echo "# Environment variables\n\
TEAM_TOKEN=your_team_token_here\n\
GEMINI_API_KEY=your_gemini_api_key_here\n\
PINECONE_API_KEY=your_pinecone_api_key_here\n\
PINECONE_ENV=your_pinecone_env_here\n\
PINECONE_INDEX=your_pinecone_index_here" > .env

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
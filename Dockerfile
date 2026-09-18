FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app package
COPY app ./app

# Expose port (default 8000, configurable via PORT env var)
EXPOSE 8000

# Set environment variable defaults
ENV PORT=8000
ENV HOST=0.0.0.0

# Start FastAPI application
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT"]

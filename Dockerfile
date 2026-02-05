# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install minimal Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU version first
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.10.0+cpu \
    torchaudio==2.10.0+cpu

# Install other dependencies from PyPI
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.32.0" \
    python-multipart==0.0.12 \
    tensorflow==2.15.0 \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    numpy \
    scipy \
    resampy \
    scikit-learn==1.5.2 \
    requests==2.32.3 \
    httpx==0.27.2 \
    python-dotenv==1.0.1 \
    pydantic==2.9.2 \
    pydantic-settings==2.5.2

# Copy application files
COPY main.py ./
COPY start.py ./
COPY models/ ./models/
COPY features/ ./features/

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE $PORT

# Run the application using Python startup script
CMD ["python", "start.py"]
# Use the maintained slim Bookworm base; pinning to slim helps reduce CVEs
FROM python:3.11-slim-bookworm AS runtime

# Install only the system libs we actually need for audio I/O
# (no build-essential in runtime). Use --no-install-recommends to keep it small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 10001 appuser
WORKDIR /app

# Copy only what's needed (avoid copying .vscode, devcontainer, node_modules, etc.)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files explicitly to avoid bundling dev artifacts that scanners flag
COPY app.py /app/app.py
COPY diarization.py /app/diarization.py
COPY features_extraction.py /app/features_extraction.py
COPY models /app/models

USER appuser

EXPOSE 8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py"]
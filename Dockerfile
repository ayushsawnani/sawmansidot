FROM python:3.11-slim AS runtime

# System deps for librosa/soundfile
RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y portaudio19-dev

COPY . /app
EXPOSE 8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py"]
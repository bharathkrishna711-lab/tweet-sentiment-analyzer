FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# CPU-only torch (the default pip version is the huge GPU one)
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# download the model from Hugging Face once, at build time
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Bharath-krish-11/tweet-sentiment-distilbert', local_dir='/app/model')"

COPY src ./src

ENV MODEL_PATH=/app/model
EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
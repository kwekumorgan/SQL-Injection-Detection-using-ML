FROM python:3.14-slim

WORKDIR /app

COPY requirements-api.txt ./requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# Keep the inference package and fitted artifacts in the same image.
COPY app ./app
COPY models ./models

ENV MODELS_DIR=/app/models
ENV SQLI_MODEL=lr_chi2
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

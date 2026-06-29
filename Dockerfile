FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

EXPOSE 8000
# Trains and registers a model on first startup if the registry is empty.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

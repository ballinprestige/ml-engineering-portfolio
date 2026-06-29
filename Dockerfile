FROM python:3.12.7-slim

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Locked, serving-only dependencies (no shap / xgboost / matplotlib / jupyter).
COPY requirements-service.txt .
RUN pip install -r requirements-service.txt

# Install the package (no path hacks) without re-pulling heavy deps, then bake an immutable,
# checksummed model into the image — training is a build-time release step, not done at serve time.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-deps -e . && python -m service.train

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
# Serves the baked artifact; fails fast if the registry is missing.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

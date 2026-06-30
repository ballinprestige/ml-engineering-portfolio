# Base image pinned by digest (python:3.12.7-slim) for a fully reproducible build.
FROM python:3.12.7-slim@sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Hash-locked, serving-only dependencies (full transitive set + hashes; no shap/xgboost/matplotlib).
COPY requirements-service.lock .
RUN pip install --require-hashes -r requirements-service.lock

# Provenance: bake the commit into the model metadata at train time. CI passes the real SHA.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

# Install the package (no path hacks, no extra deps), then bake an immutable, checksummed model
# into the image AS ROOT — training is a build-time release step, not done at serve time. The
# runtime user (appuser) gets NO write access to /app, so the baked artifact and code are genuinely
# read-only in the running container.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-deps -e . && python -m service.train

USER appuser
EXPOSE 8000
# Serves the baked, immutable artifact; fails fast if the registry is missing.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Pinned dependencies first, so code changes do not reinstall them.
COPY requirements.txt .
RUN grep -v '^-e' requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

COPY models ./models
COPY reports/test2024_predictions.parquet ./reports/
COPY app ./app
COPY .streamlit ./.streamlit

RUN useradd --create-home gamesense
USER gamesense

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "gamesense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

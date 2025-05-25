FROM python:3.11-slim

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Set Python path
ENV PYTHONPATH="/code"

# 3. Install Poetry
RUN pip install poetry==1.8.3
RUN poetry config virtualenvs.create false

WORKDIR /code

# 4. Copy dependency files
COPY pyproject.toml poetry.lock ./

# 5. Install dependencies only (not the package itself)
RUN poetry install --no-interaction --no-ansi --no-root

# 6. Copy application code
COPY ./app ./app
COPY ./packages ./packages

EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
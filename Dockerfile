FROM python:3.11-slim

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Set Python path
ENV PYTHONPATH="/code"

# 3. Install Poetry
RUN pip install poetry==1.6.1
RUN poetry config virtualenvs.create false

WORKDIR /code

# 4. Copy dependency files
COPY pyproject.toml poetry.lock README.md ./

# 5. Install dependencies ONLY (no app code yet)
RUN poetry install --no-interaction --no-ansi

# 6. Copy application code
COPY ./packages ./packages
COPY ./app ./app

# 7. Install the project as a package
RUN poetry install --no-interaction --no-ansi

EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
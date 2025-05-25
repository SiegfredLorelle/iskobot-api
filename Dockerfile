FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/code"

# Install Poetry
RUN pip install poetry==1.6.1
RUN poetry config virtualenvs.create false

WORKDIR /code

# Copy only dependency files first
COPY pyproject.toml poetry.lock* README.md ./

# Install project dependencies (without the app code)
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the code
COPY ./packages ./packages
COPY ./app ./app

# Install the project as a package
RUN poetry install --no-interaction --no-ansi

EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
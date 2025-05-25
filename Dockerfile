FROM python:3.11-slim

# Set Python path to include /code
ENV PYTHONPATH="${PYTHONPATH}:/code"

# Install Poetry
RUN pip install poetry==1.6.1
RUN poetry config virtualenvs.create false

WORKDIR /code

# Copy dependency files
COPY ./pyproject.toml ./README.md ./poetry.lock* ./

# Copy directories
COPY ./packages ./packages
COPY ./app ./app

# Install dependencies AND your app as a package
RUN poetry install --no-interaction --no-ansi

EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
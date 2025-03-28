# Iskobot Chatbot API
>  Chatbot designed to assist computer engineering students
---

## Setup and Requirements

### Software Requirements
- Python (version 3.11 or higher is required)
- LangChain CLI
- Poetry (for dependency management)
- pipx (to install and run LangChain CLI and Poetry in isolated virtual environments)

### Prerequisites
- Google Cloud project
- gcloud CLI
    ```
    gcloud auth login
    gcloud config set project <PROJECT_ID>
    ```
- if running locally, set App Default Credentials
    ```
    gcloud auth application-default login
    gcloud auth application-default set-quota-project <PROJECT_ID>
    ```
- Install dependencies
    ```
    poetry install
    ```

## Create .env file
- Create a `.env` file in your project root directory.
```
REGION=us-central1
DB_INSTANCE_NAME=your-db-instance-name
DB_USER=your-db-user
DB_PASS=your-db-password
DB_NAME=your-db-name
GCS_BUCKET_NAME=your-gcs-bucket-name
```
## Usage
### Run Locally
```bash
poetry run uvicorn app.server:app --port 8080 --reload
```

### Test API with cURL

```bash
# Sample querying when running locally
curl -X 'POST' \
  'http://localhost:8080/query' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Give me a TLDR of the paper Attention is All You need."
}'
```
```bash
# Sample querying on prod
curl -X 'POST' \
  'https://run-rag-116711660246.asia-east1.run.app/query' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Give me a TLDR of the paper Attention is All You need."
}'
```

## Experiment with Pipeline
[Iskobot Chatbot API](https://run-rag-116711660246.asia-east1.run.app)


## Deploy to Cloud Run
- Export your .env
```bash
# For example:
REGION=asia-east1
DB_INSTANCE_NAME=your-db-instance
DB_USER=your-db-user
DB_NAME=your-database-name
DB_PASS=your-secure-password
GCS_BUCKET_NAME=your-bucket-name-123
GROQ_API_KEY=your-groq-api-key-123
```

```bash
gcloud run deploy <APP_NAME> \
  --source . \
  --set-env-vars=DB_INSTANCE_NAME=$DB_INSTANCE_NAME \
  --set-env-vars=DB_USER=$DB_USER \
  --set-env-vars=DB_NAME=$DB_NAME \
  --set-env-vars=DB_PASS=$DB_PASS \
  --set-env-vars=GROQ_API_KEY=$GROQ_API_KEY \
  --region=$REGION \
  --allow-unauthenticated
```
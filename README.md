# Iskobot Chatbot API

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
## Test API with cURL
```bash
# Sample querying to chatbot running locally
curl -X 'POST' \
  'http://127.0.0.1:8080/query' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Give me a TLDR of the paper Attention is All You need."
}'
```
```bash
# Sample querying deployed chatbot
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
```
gcloud run deploy <APP_NAME> \
  --source . \
  --set-env-vars=DB_INSTANCE_NAME=$DB_INSTANCE_NAME \
  --set-env-vars=DB_USER=$DB_USER \
  --set-env-vars=DB_NAME=$DB_NAME \
  --set-env-vars=DB_PASS=$DB_PASS \
  --region=$REGION \
  --allow-unauthenticated
```
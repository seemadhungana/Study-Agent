# Study Assistant Agent

An AI-powered study assistant for AI/ML course material. Ask it to explain topics, generate quizzes, or compare concepts — it decides which tool to use each turn.

## Features

- Explain topics (transformers, RAG, agents, LLMs, etc.)
- Generate quizzes for self-testing
- Compare two concepts side by side
- Backed by RAG over course PDFs

## Stack

- FastAPI + OpenAI (GPT-4o-mini)
- AWS Lambda + API Gateway + DynamoDB (serverless deployment)
- AWS SAM for infrastructure

## Setup

**Dependencies**

```bash
pip install -r requirements.txt
```

**Environment variables**

Copy `.env.example` to `.env` and fill in the values — do not commit `.env`.

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `LLM_MODEL` | No | Model to use (default: `gpt-4o-mini`) |
| `CHAT_HISTORY_TABLE_NAME` | No | DynamoDB table name (default: `StudyAssistantSessions`) |

**Run locally**

```bash
cd app && uvicorn main:app --reload --port 8000
```

Open http://localhost:8000.

## Deployment (AWS)

Deploys as a serverless stack via AWS SAM (Lambda + API Gateway + DynamoDB).

```bash
sam build && sam deploy --guided
```

You will be prompted for your stack name, region, and `OPENAI_API_KEY`. After deploy, SAM prints the public API URL.

## Course

Context Augmented AI — Spring 2026

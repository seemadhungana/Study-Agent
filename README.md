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

## Local Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
cd app && uvicorn main:app --reload --port 8000
```

Open http://localhost:8000.

## Deploy to AWS

```bash
sam build && sam deploy --guided
```

## Course

Context Augmented AI — Spring 2026

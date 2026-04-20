# Study Assistant Agent

An agentic LLM-powered web application that helps students study AI/ML course material. The LLM decides which tools to call each turn — making it a genuine agentic system, not a fixed pipeline.

## What It Does

- **Explains topics** — retrieves facts from a structured knowledge base (transformers, LLMs, RAG, agents, context management, structured outputs, function calling)
- **Generates quizzes** — produces randomized Q&A sets for self-testing
- **Compares concepts** — side-by-side breakdowns of two topics
- **Lists available topics** — the LLM chooses to call this when the user asks what it knows

## Agentic Behavior

The LLM (GPT-4o-mini) receives tool definitions each turn and autonomously decides:
- Whether to call a tool or respond directly
- Which tool is appropriate (lookup, quiz, compare, list)
- How to synthesize tool results into an educational response

This decision happens via OpenAI function calling — the model outputs a structured tool call JSON rather than free text, and the backend dispatches it.

## Project Structure

```
final_project/
├── app/
│   ├── main.py           # FastAPI server + Mangum Lambda handler
│   ├── agent.py          # Agentic loop (LLM + tool dispatch)
│   ├── tools.py          # Tool definitions + knowledge base + quiz bank
│   ├── session_store.py  # DynamoDB-backed conversation history
│   ├── observability.py  # Structured logging + metrics
│   ├── logs/             # JSONL trace logs (local only)
│   └── static/
│       └── index.html    # Frontend UI
├── template.yaml         # AWS SAM deployment template
├── requirements.txt
├── .env.example
└── README.md
```

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum
```

### 3. Run locally

```bash
cd app
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

> **Note:** Running locally uses the DynamoDB session store by default. Either set up AWS credentials (`aws configure`) pointing at a real DynamoDB table, or see the local fallback note below.

### Local fallback without AWS

If you want to run fully locally without AWS, temporarily revert `main.py` to use an in-memory dict for sessions:

```python
# In main.py, replace get_history/save_history calls with:
_sessions = {}
history = _sessions.get(session_id, [])
...
_sessions[session_id] = history
```

---

## AWS Deployment (Lambda + API Gateway + DynamoDB)

This app deploys as a **serverless** stack using [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html).

### Architecture

```
Browser → API Gateway (HTTP API) → Lambda (FastAPI + Mangum) → OpenAI API
                                          ↕
                                     DynamoDB (session history)
```

### Prerequisites

1. [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) configured (`aws configure`)
2. [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) installed
3. An S3 bucket for SAM deployment artifacts (SAM can create one for you)

### Deploy

```bash
# From the final_project/ directory:

# Build — packages the app/ directory and its dependencies into a Lambda zip
sam build

# Deploy — creates/updates all AWS resources
sam deploy --guided
```

During `sam deploy --guided` you will be prompted for:

| Prompt | Value |
|---|---|
| Stack name | `study-assistant-agent` (or any name) |
| AWS Region | e.g. `us-east-1` |
| OpenAIApiKey | Your OpenAI API key (stored as a parameter, never in code) |
| Confirm changeset | `y` |
| Allow SAM to create IAM roles | `y` |
| Save arguments to samconfig.toml | `y` (makes future deploys just `sam deploy`) |

After deploy completes, SAM prints the **ApiUrl** output — that is your public URL.

### Subsequent deploys

```bash
sam build && sam deploy
```

### What SAM creates

| Resource | Details |
|---|---|
| **Lambda function** | `StudyAssistantAgent` — Python 3.12, 512 MB, 30s timeout |
| **API Gateway** | HTTP API v2 — proxies all paths/methods to Lambda |
| **DynamoDB table** | `StudyAssistantSessions` — PAY_PER_REQUEST, TTL enabled (7-day session expiry) |
| **IAM role** | Least-privilege: Lambda can only read/write/delete its own DynamoDB table |

### Tear down

```bash
sam delete --stack-name study-assistant-agent
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model for the agent |
| `CHAT_HISTORY_TABLE_NAME` | No | `StudyAssistantSessions` | DynamoDB table name (set automatically by SAM) |

**Never commit `.env` or your API key to version control.**

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve the frontend |
| `POST` | `/chat` | Submit a message, get an agent response |
| `GET` | `/metrics` | Live observability metrics |
| `GET` | `/traces` | Recent JSONL trace records |
| `DELETE` | `/session/{id}` | Clear a conversation session |

### POST /chat

Request:
```json
{ "message": "Quiz me on transformers", "session_id": "optional-uuid" }
```

Response:
```json
{
  "response": "Here are 3 questions on transformers...",
  "session_id": "abc-123",
  "trace_id": "f4a1b2c3",
  "tool_calls_made": 1,
  "latency_ms": 843.2
}
```

---

## Observability

All requests are logged as JSONL to `app/logs/agent_traces.jsonl` locally (on Lambda, logs go to CloudWatch). Each record captures:
- `trace_id` — links all events for one request
- `event` — `request_start`, `tool_call`, or `response_complete`
- `tool_name`, `arguments`, `success` for tool calls
- `latency_ms`, `error`

Metrics at `GET /metrics`:
- **Tool call success rate** — quality/usefulness metric
- **Average response latency** — operational metric

On Lambda, structured logs are viewable in **CloudWatch Logs** under `/aws/lambda/StudyAssistantAgent`.

---

## Estimated AWS Cost (demo/course usage)

| Service | Free Tier | Est. cost beyond free tier |
|---|---|---|
| Lambda | 1M requests/month free | ~$0.20 per million requests |
| API Gateway (HTTP) | 1M requests/month free | ~$1.00 per million requests |
| DynamoDB | 25 GB storage + 200M requests free | Negligible at demo scale |
| **Total** | **~$0/month** at course demo traffic | |

---

## Acknowledgments

- OpenAI Python SDK for function calling
- FastAPI + Mangum for Lambda-compatible ASGI
- AWS SAM for infrastructure as code
- Course materials from the Context Augmented AI course (Spring 2026)

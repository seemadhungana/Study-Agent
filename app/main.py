"""
FastAPI backend for the Study Assistant Agent.
Runs locally via uvicorn OR on AWS Lambda via the Mangum adapter.

Endpoints:
  GET  /            → serve frontend HTML
  POST /chat        → agentic chat turn
  GET  /metrics     → observability metrics
  GET  /traces      → recent agent traces
  DELETE /session/{session_id} → clear a conversation
"""

import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from mangum import Mangum
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import run_agent
from observability import (
    get_metrics,
    get_recent_traces,
    log_request_start,
    log_response_complete,
)
from session_store import clear_history, get_history, save_history

load_dotenv()

app = FastAPI(title="Study Assistant Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bedrock model ID — override via LLM_MODEL env var in template.yaml if needed
MODEL = os.environ.get("LLM_MODEL", "anthropic.claude-haiku-4-5-20251001-v1:0")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.get("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    user_message = req.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    trace_id = log_request_start(session_id, user_message)
    start = time.perf_counter()
    error_msg = None

    try:
        # Load history from DynamoDB (persistent across Lambda invocations)
        history = get_history(session_id)

        response_text, tool_calls_made = run_agent(
            user_message=user_message,
            conversation_history=history,
            trace_id=trace_id,
            model=MODEL,
        )

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response_text})

        # Persist updated history to DynamoDB
        save_history(session_id, history)

    except Exception as e:
        error_msg = str(e)
        response_text = "I encountered an error. Please try again."
        tool_calls_made = 0

    latency_ms = (time.perf_counter() - start) * 1000
    log_response_complete(trace_id, session_id, response_text, latency_ms, tool_calls_made, error_msg)

    if error_msg:
        raise HTTPException(status_code=500, detail=error_msg)

    return JSONResponse({
        "response": response_text,
        "session_id": session_id,
        "trace_id": trace_id,
        "tool_calls_made": tool_calls_made,
        "latency_ms": round(latency_ms, 1),
    })


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    clear_history(session_id)
    return {"message": f"Session {session_id} cleared."}


@app.get("/metrics")
def metrics():
    return JSONResponse(get_metrics())


@app.get("/traces")
def traces(n: int = 20):
    return JSONResponse(get_recent_traces(n))


# Lambda entry point — Mangum wraps the FastAPI app for API Gateway
handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

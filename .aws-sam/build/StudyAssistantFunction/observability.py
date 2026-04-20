"""
Observability layer: structured logging + in-memory metrics.

Captures: user inputs, model outputs, tool calls, latency, errors.
Metrics tracked:
  1. tool_call_success_rate  — quality / usefulness proxy
  2. avg_response_latency_ms — operational health metric
"""

import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Lambda's filesystem is read-only except /tmp; use that when running on AWS
_lambda_tmp = Path("/tmp")
_local_logs = Path(__file__).parent / "logs"
LOG_DIR = _lambda_tmp if _lambda_tmp.exists() and not _local_logs.exists() else _local_logs
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "agent_traces.jsonl"

_metrics: dict = {
    "total_requests": 0,
    "total_tool_calls": 0,
    "successful_tool_calls": 0,
    "failed_tool_calls": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
    "tool_call_counts": defaultdict(int),
}


def _write_log(record: dict) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_request_start(session_id: str, user_message: str) -> str:
    trace_id = str(uuid.uuid4())[:8]
    _metrics["total_requests"] += 1
    record = {
        "trace_id": trace_id,
        "session_id": session_id,
        "event": "request_start",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message": user_message,
    }
    _write_log(record)
    return trace_id


def log_tool_call(trace_id: str, tool_name: str, arguments: dict, result: str, success: bool) -> None:
    _metrics["total_tool_calls"] += 1
    _metrics["tool_call_counts"][tool_name] += 1
    if success:
        _metrics["successful_tool_calls"] += 1
    else:
        _metrics["failed_tool_calls"] += 1

    record = {
        "trace_id": trace_id,
        "event": "tool_call",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "arguments": arguments,
        "result_preview": result[:300],
        "success": success,
    }
    _write_log(record)


def log_response_complete(
    trace_id: str,
    session_id: str,
    assistant_response: str,
    latency_ms: float,
    tool_calls_made: int,
    error: Optional[str] = None,
) -> None:
    _metrics["total_latency_ms"] += latency_ms
    if error:
        _metrics["errors"] += 1

    record = {
        "trace_id": trace_id,
        "session_id": session_id,
        "event": "response_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "assistant_response_preview": assistant_response[:300],
        "latency_ms": round(latency_ms, 2),
        "tool_calls_made": tool_calls_made,
        "error": error,
    }
    _write_log(record)


def get_metrics() -> dict:
    total = _metrics["total_requests"]
    tool_total = _metrics["total_tool_calls"]
    avg_latency = (
        round(_metrics["total_latency_ms"] / total, 1) if total > 0 else 0.0
    )
    tool_success_rate = (
        round(_metrics["successful_tool_calls"] / tool_total * 100, 1)
        if tool_total > 0
        else None
    )
    return {
        "total_requests": total,
        "total_tool_calls": tool_total,
        "tool_call_success_rate_pct": tool_success_rate,
        "avg_response_latency_ms": avg_latency,
        "errors": _metrics["errors"],
        "tool_call_breakdown": dict(_metrics["tool_call_counts"]),
    }


def get_recent_traces(n: int = 20) -> list:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text().strip().splitlines()
    recent = lines[-n:] if len(lines) > n else lines
    records = []
    for line in reversed(recent):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records

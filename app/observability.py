"""
Observability layer: structured logging + DynamoDB-backed metrics.

Metrics persist across all Lambda invocations via atomic DynamoDB counters.
Logs are written to /tmp as JSONL for trace inspection.

Metrics tracked:
  1. tool_call_success_rate  — quality / usefulness proxy
  2. avg_response_latency_ms — operational health metric
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

# JSONL log for traces (ephemeral per Lambda container, fine for debugging)
_lambda_tmp = Path("/tmp")
_local_logs = Path(__file__).parent / "logs"
LOG_DIR = _lambda_tmp if _lambda_tmp.exists() and not _local_logs.exists() else _local_logs
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "agent_traces.jsonl"

# DynamoDB config — reuse the same table, metrics stored under PK=METRICS
import os
TABLE_NAME = os.environ.get("CHAT_HISTORY_TABLE_NAME", "StudyAssistantSessions")
METRICS_PK = "METRICS"
METRICS_SK = "GLOBAL"

_dynamodb = None


def _get_table():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb").Table(TABLE_NAME)
    return _dynamodb


def _write_log(record: dict) -> None:
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _increment_metrics(latency_ms: float, tool_calls: int, successful_tools: int, failed_tools: int, error: bool):
    """Atomically increment all metric counters in DynamoDB."""
    table = _get_table()
    try:
        table.update_item(
            Key={"PK": METRICS_PK, "SK": METRICS_SK},
            UpdateExpression=(
                "ADD total_requests :one, "
                "total_tool_calls :tc, "
                "successful_tool_calls :stc, "
                "failed_tool_calls :ftc, "
                "total_latency_ms :lat, "
                "errors :err"
            ),
            ExpressionAttributeValues={
                ":one": 1,
                ":tc": tool_calls,
                ":stc": successful_tools,
                ":ftc": failed_tools,
                ":lat": int(latency_ms),
                ":err": 1 if error else 0,
            },
        )
    except ClientError:
        pass


def _increment_tool_count(tool_name: str, success: bool):
    """Atomically increment per-tool call counter and success/fail."""
    table = _get_table()
    try:
        table.update_item(
            Key={"PK": METRICS_PK, "SK": METRICS_SK},
            UpdateExpression=(
                "ADD #tc :one, #ts :s"
            ),
            ExpressionAttributeNames={
                "#tc": f"tool_{tool_name}_calls",
                "#ts": f"tool_{tool_name}_success",
            },
            ExpressionAttributeValues={
                ":one": 1,
                ":s": 1 if success else 0,
            },
        )
    except ClientError:
        pass


def log_request_start(session_id: str, user_message: str) -> str:
    trace_id = str(uuid.uuid4())[:8]
    _write_log({
        "trace_id": trace_id,
        "session_id": session_id,
        "event": "request_start",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message": user_message,
    })
    return trace_id


def log_tool_call(trace_id: str, tool_name: str, arguments: dict, result: str, success: bool) -> None:
    _write_log({
        "trace_id": trace_id,
        "event": "tool_call",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "arguments": arguments,
        "result_preview": result[:300],
        "success": success,
    })
    _increment_tool_count(tool_name, success)


def log_response_complete(
    trace_id: str,
    session_id: str,
    assistant_response: str,
    latency_ms: float,
    tool_calls_made: int,
    successful_tools: int = 0,
    error: Optional[str] = None,
) -> None:
    failed_tools = tool_calls_made - successful_tools
    _write_log({
        "trace_id": trace_id,
        "session_id": session_id,
        "event": "response_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "assistant_response_preview": assistant_response[:300],
        "latency_ms": round(latency_ms, 2),
        "tool_calls_made": tool_calls_made,
        "error": error,
    })
    _increment_metrics(latency_ms, tool_calls_made, successful_tools, failed_tools, bool(error))


def get_metrics() -> dict:
    table = _get_table()
    try:
        resp = table.get_item(Key={"PK": METRICS_PK, "SK": METRICS_SK})
        item = resp.get("Item", {})
    except ClientError:
        item = {}

    total = int(item.get("total_requests", 0))
    tool_total = int(item.get("total_tool_calls", 0))
    successful = int(item.get("successful_tool_calls", 0))
    total_latency = int(item.get("total_latency_ms", 0))
    errors = int(item.get("errors", 0))

    avg_latency = round(total_latency / total, 1) if total > 0 else 0.0
    tool_success_rate = round(successful / tool_total * 100, 1) if tool_total > 0 else None

    # Collect per-tool breakdown from dynamic keys
    tool_counts = {}
    last_tool = None
    for key, val in item.items():
        if key.startswith("tool_") and key.endswith("_calls"):
            name = key[len("tool_"):-len("_calls")]
            tool_counts[name] = int(val)
            last_tool = name

    return {
        "total_requests": total,
        "total_tool_calls": tool_total,
        "tool_call_success_rate_pct": tool_success_rate,
        "avg_response_latency_ms": avg_latency,
        "errors": errors,
        "tool_call_breakdown": tool_counts,
        "last_tool_called": last_tool,
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

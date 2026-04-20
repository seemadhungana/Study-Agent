"""
DynamoDB-backed session store for conversation history.

Table schema (PAY_PER_REQUEST billing):
  PK  (String) — "SESSION#<session_id>"
  SK  (String) — "HISTORY"
  messages (List) — [{"role": "user"|"assistant", "content": "..."}]
  ttl  (Number) — Unix timestamp; DynamoDB auto-deletes expired items
"""

import json
import logging
import os
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get("CHAT_HISTORY_TABLE_NAME", "StudyAssistantSessions")
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days
MAX_HISTORY_MESSAGES = 20


def _table():
    return boto3.resource("dynamodb").Table(TABLE_NAME)


def get_history(session_id: str) -> list:
    """Return conversation history for a session (list of {role, content} dicts)."""
    try:
        response = _table().get_item(
            Key={"PK": f"SESSION#{session_id}", "SK": "HISTORY"}
        )
        item = response.get("Item")
        if item:
            return item.get("messages", [])
    except ClientError as e:
        logger.error("DynamoDB get_history error: %s", e)
    return []


def save_history(session_id: str, messages: list) -> None:
    """Persist conversation history, capped to MAX_HISTORY_MESSAGES."""
    if len(messages) > MAX_HISTORY_MESSAGES:
        messages = messages[-MAX_HISTORY_MESSAGES:]
    expire_at = int(time.time()) + SESSION_TTL_SECONDS
    try:
        _table().put_item(
            Item={
                "PK": f"SESSION#{session_id}",
                "SK": "HISTORY",
                "messages": messages,
                "ttl": expire_at,
            }
        )
    except ClientError as e:
        logger.error("DynamoDB save_history error: %s", e)


def clear_history(session_id: str) -> None:
    """Delete a session's history from DynamoDB."""
    try:
        _table().delete_item(
            Key={"PK": f"SESSION#{session_id}", "SK": "HISTORY"}
        )
    except ClientError as e:
        logger.error("DynamoDB clear_history error: %s", e)


def ensure_table_exists() -> None:
    """Create the DynamoDB table if it doesn't exist (idempotent)."""
    dynamodb = boto3.resource("dynamodb")
    try:
        dynamodb.meta.client.describe_table(TableName=TABLE_NAME)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            logger.info("Creating DynamoDB table: %s", TABLE_NAME)
            table = dynamodb.create_table(
                TableName=TABLE_NAME,
                KeySchema=[
                    {"AttributeName": "PK", "KeyType": "HASH"},
                    {"AttributeName": "SK", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "PK", "AttributeType": "S"},
                    {"AttributeName": "SK", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.meta.client.get_waiter("table_exists").wait(TableName=TABLE_NAME)
            logger.info("Table created: %s", TABLE_NAME)
        else:
            raise

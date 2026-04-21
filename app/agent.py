"""
Agentic LLM loop using Amazon Bedrock (Claude Haiku) via the Converse API.

The Bedrock Converse API supports tool use natively with the same
request/response shape across all Claude models, so no OpenAI SDK needed.

The LLM decides each turn whether to:
  - Call a tool (lookup_topic, generate_quiz, list_topics, compare_topics)
  - Return a final answer directly

This is the core agentic behavior: the model controls workflow via tool selection.
"""

import json
import boto3

from tools import TOOL_DEFINITIONS, dispatch_tool
from observability import log_tool_call

# Claude Haiku on Bedrock — fast and cheap, strong tool-use support.
# Override with the LLM_MODEL env var in template.yaml if needed.
DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"

SYSTEM_PROMPT = """You are a helpful Study Assistant for an AI/ML course covering transformers, \
large language models, RAG, agents, context management, structured outputs, and function calling.

You have access to a knowledge base via tools. Use them when the user asks about a specific topic, \
wants to be quizzed, wants to compare concepts, or asks what topics are available.

Guidelines:
- If the user asks about a topic covered in the knowledge base, ALWAYS call lookup_topic first to retrieve accurate facts.
- If the user wants a quiz or practice questions, call generate_quiz.
- If the user asks what topics are available, call list_topics.
- If the user wants to compare two concepts, call compare_topics.
- For general conversation or questions outside the knowledge base, respond directly without a tool call.
- After receiving tool results, synthesize them into a clear, friendly, educational response.
- Never make up facts — rely on what the tools return.
"""

# Bedrock expects tool definitions in its own schema (slightly different from OpenAI).
# Convert our existing TOOL_DEFINITIONS list once at import time.
def _to_bedrock_tools(openai_tools: list) -> list:
    bedrock_tools = []
    for t in openai_tools:
        fn = t["function"]
        bedrock_tools.append({
            "toolSpec": {
                "name": fn["name"],
                "description": fn["description"],
                "inputSchema": {
                    "json": fn["parameters"]
                }
            }
        })
    return bedrock_tools

BEDROCK_TOOL_DEFINITIONS = _to_bedrock_tools(TOOL_DEFINITIONS)


def _get_bedrock_client():
    # boto3 uses the Lambda execution role automatically when running on AWS.
    # Locally it uses ~/.aws/credentials.
    return boto3.client("bedrock-runtime")


def run_agent(
    user_message: str,
    conversation_history: list,
    trace_id: str,
    model: str = DEFAULT_MODEL,
    max_iterations: int = 5,
    client=None,          # kept for API compatibility; ignored (Bedrock uses boto3)
) -> tuple[str, int]:
    """
    Run the agentic loop for one user turn using Bedrock Converse API.

    conversation_history: list of {"role": "user"|"assistant", "content": "..."}
    Returns (assistant_response_text, num_tool_calls_made).
    """
    bedrock = _get_bedrock_client()

    # Build Bedrock message list from stored history + new user turn.
    # Bedrock Converse expects: [{"role": ..., "content": [{"text": ...}]}, ...]
    messages = []
    for msg in conversation_history:
        messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}],
        })
    messages.append({"role": "user", "content": [{"text": user_message}]})

    tool_calls_made = 0
    successful_tool_calls = 0

    for _ in range(max_iterations):
        response = bedrock.converse(
            modelId=model,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig={
                "tools": BEDROCK_TOOL_DEFINITIONS,
                "toolChoice": {"auto": {}},
            },
        )

        stop_reason = response["stopReason"]
        output_message = response["output"]["message"]
        messages.append(output_message)

        # "end_turn" means the model is done — no tool call, return text.
        if stop_reason == "end_turn":
            text = ""
            for block in output_message.get("content", []):
                if "text" in block:
                    text += block["text"]
            return text, tool_calls_made, successful_tool_calls

        # "tool_use" means the model wants to call one or more tools.
        if stop_reason == "tool_use":
            tool_results = []
            for block in output_message.get("content", []):
                if "toolUse" not in block:
                    continue

                tool_use = block["toolUse"]
                tool_name = tool_use["name"]
                tool_use_id = tool_use["toolUseId"]
                arguments = tool_use.get("input", {})

                result_str = dispatch_tool(tool_name, arguments)

                try:
                    parsed = json.loads(result_str)
                    success = "error" not in parsed and parsed.get("found", True) is not False
                except Exception:
                    success = False

                log_tool_call(trace_id, tool_name, arguments, result_str, success)
                tool_calls_made += 1
                if success:
                    successful_tool_calls += 1

                tool_results.append({
                    "toolUseId": tool_use_id,
                    "content": [{"text": result_str}],
                })

            # Feed all tool results back as a single "user" turn (Bedrock convention).
            messages.append({
                "role": "user",
                "content": [{"toolResult": tr} for tr in tool_results],
            })
            continue

        # Any other stop reason — just extract whatever text is there.
        text = ""
        for block in output_message.get("content", []):
            if "text" in block:
                text += block["text"]
        return text, tool_calls_made, successful_tool_calls

    # Fallback: hit max_iterations, ask for a plain final answer.
    response = bedrock.converse(
        modelId=model,
        system=[{"text": SYSTEM_PROMPT}],
        messages=messages,
    )
    text = ""
    for block in response["output"]["message"].get("content", []):
        if "text" in block:
            text += block["text"]
    return text, tool_calls_made, successful_tool_calls

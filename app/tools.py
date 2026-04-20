"""
Tool implementations for the Study Assistant Agent.
Each function here maps to a tool the LLM can call.
"""

import json
import random
from datetime import datetime


TOPIC_KNOWLEDGE_BASE = {
    "transformers": {
        "summary": "Transformers are a neural network architecture based entirely on attention mechanisms, introduced in 'Attention Is All You Need' (Vaswani et al., 2017). They replaced RNNs for most NLP tasks by enabling parallel processing of sequences.",
        "key_concepts": ["self-attention", "multi-head attention", "positional encoding", "encoder-decoder", "feed-forward layers"],
        "facts": [
            "Transformers use scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V",
            "BERT uses only the encoder stack; GPT uses only the decoder stack.",
            "Positional encodings inject sequence order since attention is order-agnostic.",
            "Multi-head attention runs several attention heads in parallel to capture different relationships.",
        ],
    },
    "llm": {
        "summary": "Large Language Models (LLMs) are transformer-based models trained on massive text corpora to predict the next token. They exhibit emergent capabilities like reasoning, code generation, and instruction following.",
        "key_concepts": ["pre-training", "fine-tuning", "RLHF", "in-context learning", "prompt engineering", "temperature", "top-p sampling"],
        "facts": [
            "GPT-3 has 175 billion parameters.",
            "In-context learning lets LLMs solve tasks from a few examples in the prompt without weight updates.",
            "RLHF (Reinforcement Learning from Human Feedback) aligns models to human preferences.",
            "Temperature controls randomness: 0 = greedy decoding, >1 = more random.",
        ],
    },
    "rag": {
        "summary": "Retrieval-Augmented Generation (RAG) combines a retrieval system (usually vector search) with an LLM. The retriever fetches relevant documents; the LLM generates an answer grounded in those documents.",
        "key_concepts": ["vector embeddings", "semantic search", "chunking", "re-ranking", "grounding", "hallucination reduction"],
        "facts": [
            "RAG reduces hallucinations by grounding the LLM in retrieved evidence.",
            "Documents are chunked and embedded; queries are embedded and matched by cosine similarity.",
            "Re-rankers (e.g., cross-encoders) improve precision after initial retrieval.",
            "Hybrid search combines dense (embedding) and sparse (BM25) retrieval.",
        ],
    },
    "agents": {
        "summary": "An AI agent is an LLM that can decide what to do next—calling tools, routing to sub-systems, or producing a final answer. The LLM acts as the 'brain' that plans and executes steps.",
        "key_concepts": ["tool use", "function calling", "ReAct", "chain-of-thought", "multi-agent", "supervisor pattern"],
        "facts": [
            "ReAct (Reason + Act) interleaves reasoning traces with tool calls.",
            "Function calling lets LLMs output structured JSON specifying which tool to invoke and with what arguments.",
            "Multi-agent systems use a supervisor LLM to route tasks to specialized sub-agents.",
            "Tool selection is where the agentic decision happens: the LLM chooses which (if any) tool to call.",
        ],
    },
    "context management": {
        "summary": "Context management deals with what information the LLM receives in its context window. Strategies include summarization, sliding windows, memory stores, and dynamic retrieval to stay within token limits.",
        "key_concepts": ["context window", "token limit", "summarization", "sliding window", "external memory", "KV cache"],
        "facts": [
            "Modern LLMs have context windows ranging from 8K to 1M+ tokens.",
            "Lost-in-the-middle effect: LLMs attend better to information at the beginning and end of long contexts.",
            "Summarization compresses past turns to save tokens for new content.",
            "KV-cache stores computed attention keys/values so repeated prefixes need not be recomputed.",
        ],
    },
    "structured outputs": {
        "summary": "Structured outputs constrain LLM responses to a defined schema (JSON, XML, etc.), making them reliable for downstream processing. Techniques include JSON mode, grammar-constrained decoding, and function calling.",
        "key_concepts": ["JSON mode", "function calling", "Pydantic", "grammar-constrained decoding", "schema validation"],
        "facts": [
            "OpenAI's JSON mode guarantees the response is valid JSON.",
            "Pydantic models can be auto-converted to JSON schemas for structured output validation.",
            "Grammar-constrained decoding (e.g., llama.cpp GBNF) restricts the token distribution at each step.",
            "Structured outputs eliminate the need for regex-based output parsing.",
        ],
    },
    "function calling": {
        "summary": "Function calling (tool use) is a model capability where the LLM outputs a structured call to a predefined function instead of free text. The caller executes the function and feeds results back to the model.",
        "key_concepts": ["tool schema", "tool result", "parallel tool calls", "tool choice", "streaming tool calls"],
        "facts": [
            "Tool schemas are defined as JSON Schema objects describing name, description, and parameters.",
            "The LLM does not execute functions—it just outputs the call; the application runs it.",
            "Parallel tool calls allow the model to invoke multiple tools in one turn.",
            "tool_choice='required' forces the model to always call a tool.",
        ],
    },
}

QUIZ_QUESTIONS = {
    "transformers": [
        {"q": "What is the core mechanism in transformers that replaces recurrence?", "a": "Self-attention (scaled dot-product attention)"},
        {"q": "What does 'multi-head' mean in multi-head attention?", "a": "Multiple attention heads run in parallel, each learning different relationships"},
        {"q": "Why are positional encodings needed in transformers?", "a": "Because attention is order-agnostic; positional encodings inject sequence position information"},
        {"q": "Which transformer variant uses only the encoder stack?", "a": "BERT"},
        {"q": "What is the formula for scaled dot-product attention?", "a": "softmax(QK^T / sqrt(d_k)) * V"},
    ],
    "llm": [
        {"q": "What does RLHF stand for and why is it used?", "a": "Reinforcement Learning from Human Feedback — used to align LLMs with human preferences"},
        {"q": "What is in-context learning?", "a": "Solving tasks from examples in the prompt without updating model weights"},
        {"q": "What effect does temperature=0 have on generation?", "a": "Greedy decoding — always picks the highest probability token"},
        {"q": "Approximately how many parameters does GPT-3 have?", "a": "175 billion"},
    ],
    "rag": [
        {"q": "What problem does RAG primarily address?", "a": "Hallucination — it grounds the LLM in retrieved evidence"},
        {"q": "What similarity metric is most common in vector search?", "a": "Cosine similarity"},
        {"q": "What is chunking in RAG?", "a": "Splitting documents into smaller pieces before embedding them"},
        {"q": "What does a re-ranker do in a RAG pipeline?", "a": "Re-orders retrieved documents by relevance to improve precision"},
    ],
    "agents": [
        {"q": "What does ReAct stand for?", "a": "Reason + Act — interleaving reasoning with tool calls"},
        {"q": "What makes a system 'agentic'?", "a": "The LLM makes decisions about what to do next, including which tools to call"},
        {"q": "In function calling, who executes the function?", "a": "The application/caller — the LLM only outputs the call"},
        {"q": "What is the supervisor pattern in multi-agent systems?", "a": "A supervisor LLM routes tasks to specialized sub-agents"},
    ],
    "context management": [
        {"q": "What is the lost-in-the-middle effect?", "a": "LLMs attend better to info at the beginning and end of long contexts, missing middle content"},
        {"q": "What is a KV cache?", "a": "Cached attention keys/values so repeated prefixes don't need recomputation"},
        {"q": "Name two strategies for managing long conversations within token limits.", "a": "Summarization and sliding window (also: external memory, RAG)"},
    ],
}


def lookup_topic(topic: str) -> dict:
    """Retrieve factual information about a study topic from the knowledge base."""
    topic_lower = topic.lower().strip()
    matched_key = None
    for key in TOPIC_KNOWLEDGE_BASE:
        if key in topic_lower or topic_lower in key:
            matched_key = key
            break

    if matched_key is None:
        return {
            "found": False,
            "topic": topic,
            "message": f"No knowledge base entry found for '{topic}'. Available topics: {', '.join(TOPIC_KNOWLEDGE_BASE.keys())}",
        }

    data = TOPIC_KNOWLEDGE_BASE[matched_key]
    return {
        "found": True,
        "topic": matched_key,
        "summary": data["summary"],
        "key_concepts": data["key_concepts"],
        "facts": data["facts"],
    }


def generate_quiz(topic: str, num_questions: int = 3) -> dict:
    """Generate a short quiz on a given topic."""
    topic_lower = topic.lower().strip()
    matched_key = None
    for key in QUIZ_QUESTIONS:
        if key in topic_lower or topic_lower in key:
            matched_key = key
            break

    if matched_key is None:
        available = ", ".join(QUIZ_QUESTIONS.keys())
        return {
            "found": False,
            "topic": topic,
            "message": f"No quiz available for '{topic}'. Available topics: {available}",
        }

    pool = QUIZ_QUESTIONS[matched_key]
    selected = random.sample(pool, min(num_questions, len(pool)))
    return {
        "found": True,
        "topic": matched_key,
        "num_questions": len(selected),
        "questions": [{"number": i + 1, "question": q["q"], "answer": q["a"]} for i, q in enumerate(selected)],
    }


def list_topics() -> dict:
    """List all topics available in the knowledge base."""
    return {
        "available_topics": list(TOPIC_KNOWLEDGE_BASE.keys()),
        "count": len(TOPIC_KNOWLEDGE_BASE),
        "description": "These are the topics I can look up, quiz you on, or explain in depth.",
    }


def compare_topics(topic_a: str, topic_b: str) -> dict:
    """Compare two topics side-by-side based on knowledge base entries."""
    result_a = lookup_topic(topic_a)
    result_b = lookup_topic(topic_b)

    if not result_a["found"] or not result_b["found"]:
        missing = []
        if not result_a["found"]:
            missing.append(topic_a)
        if not result_b["found"]:
            missing.append(topic_b)
        return {"found": False, "message": f"Could not find topics: {', '.join(missing)}"}

    return {
        "found": True,
        "topic_a": {
            "name": result_a["topic"],
            "summary": result_a["summary"],
            "key_concepts": result_a["key_concepts"],
        },
        "topic_b": {
            "name": result_b["topic"],
            "summary": result_b["summary"],
            "key_concepts": result_b["key_concepts"],
        },
    }


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_topic",
            "description": (
                "Look up factual information about a specific study topic from the knowledge base. "
                "Use this when the user asks about a concept, wants a summary, or needs facts about a topic. "
                "Available topics include: transformers, llm, rag, agents, context management, structured outputs, function calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The study topic to look up (e.g. 'transformers', 'RAG', 'agents').",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": (
                "Generate a short quiz with questions and answers on a given topic. "
                "Use this when the user wants to be quizzed, tested, or practice questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to quiz on.",
                    },
                    "num_questions": {
                        "type": "integer",
                        "description": "How many questions to generate (default 3, max 5).",
                        "default": 3,
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_topics",
            "description": "List all topics available in the knowledge base. Use when the user asks what topics are available or what they can study.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_topics",
            "description": "Compare two topics side-by-side. Use when the user wants to understand the difference or relationship between two concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_a": {"type": "string", "description": "First topic."},
                    "topic_b": {"type": "string", "description": "Second topic."},
                },
                "required": ["topic_a", "topic_b"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "lookup_topic": lookup_topic,
    "generate_quiz": generate_quiz,
    "list_topics": list_topics,
    "compare_topics": compare_topics,
}


def dispatch_tool(name: str, arguments: dict) -> str:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    result = fn(**arguments)
    return json.dumps(result)

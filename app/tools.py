"""
Tool implementations for the Study Assistant Agent.

Knowledge base is grounded in two academic papers:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - "Formal Algorithms for Transformers" (Phuong & Hutter, DeepMind)

Retrieval uses ChromaDB + Bedrock Titan embeddings stored in S3.
"""

import json
import os
import random

# Fallback quiz bank (always available; supplements retrieved content)
QUIZ_BANK = {
    "attention": [
        {"q": "What is the scaled dot-product attention formula?", "a": "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V"},
        {"q": "Why do we scale by sqrt(d_k) in attention?", "a": "To prevent dot products from growing large in magnitude and pushing softmax into regions of small gradients"},
        {"q": "What is multi-head attention?", "a": "Running h parallel attention heads on projected Q, K, V then concatenating and projecting the outputs"},
        {"q": "What does self-attention compute?", "a": "A weighted average of values where weights come from the compatibility of each query with all keys in the same sequence"},
        {"q": "What is the complexity of self-attention vs recurrence?", "a": "Self-attention is O(n^2 * d) per layer; recurrence is O(n * d^2). Attention is faster for long sequences."},
    ],
    "transformers": [
        {"q": "What architecture did the transformer replace for sequence-to-sequence tasks?", "a": "Recurrent Neural Networks (RNNs) and LSTMs with encoder-decoder structure"},
        {"q": "Why are positional encodings needed?", "a": "Attention has no inherent notion of order; positional encodings inject position information using sine/cosine functions"},
        {"q": "What is the role of the encoder in a transformer?", "a": "Maps input token sequence to continuous representations (keys and values) used by the decoder"},
        {"q": "What is the role of the decoder in a transformer?", "a": "Auto-regressively generates the output sequence, attending to encoder output and previous decoder outputs"},
        {"q": "What is residual connection + layer norm used for in transformers?", "a": "Stabilizes training by adding the input of each sub-layer to its output before normalization"},
        {"q": "How many attention heads and model dimension did the original transformer use?", "a": "8 heads, d_model=512 in the base model; 16 heads, d_model=1024 in the large model"},
    ],
    "training": [
        {"q": "What optimizer was used to train the original transformer?", "a": "Adam with a custom learning rate schedule: warmup then inverse square root decay"},
        {"q": "What regularization techniques did the original transformer use?", "a": "Residual dropout (P=0.1), label smoothing (epsilon=0.1)"},
        {"q": "What is label smoothing and why is it used?", "a": "Distributing a small probability mass to non-target tokens; improves generalization even though it hurts perplexity"},
        {"q": "What task was the original transformer evaluated on?", "a": "Machine translation: WMT 2014 English-German and English-French"},
        {"q": "What BLEU score did the big transformer achieve on EN-DE?", "a": "28.4 BLEU, outperforming all previous models including ensembles"},
    ],
    "bert": [
        {"q": "What is BERT's architecture?", "a": "Encoder-only transformer, trained with masked language modeling (MLM) and next sentence prediction (NSP)"},
        {"q": "What is masked language modeling?", "a": "Randomly masking 15% of input tokens and training the model to predict them — enables bidirectional context"},
        {"q": "How does BERT differ from GPT architecturally?", "a": "BERT uses encoder-only (bidirectional); GPT uses decoder-only (causal/unidirectional)"},
    ],
    "gpt": [
        {"q": "What is causal (autoregressive) language modeling?", "a": "Predicting the next token given all previous tokens; used in GPT-style decoder-only models"},
        {"q": "What masking does GPT use in attention?", "a": "Causal mask: each position can only attend to itself and previous positions, not future ones"},
        {"q": "What is the key difference between GPT pre-training and fine-tuning?", "a": "Pre-training is unsupervised next-token prediction on large corpus; fine-tuning adapts to a supervised task"},
    ],
}

AVAILABLE_TOPICS = [
    "attention mechanism",
    "transformer architecture",
    "multi-head attention",
    "positional encoding",
    "encoder-decoder",
    "self-attention",
    "BERT",
    "GPT",
    "training transformers",
    "formal algorithms",
    "scaled dot-product attention",
]

_vector_store_available = None


def _check_vector_store():
    global _vector_store_available
    if _vector_store_available is not None:
        return _vector_store_available
    bucket = os.environ.get("KNOWLEDGE_BUCKET", "")
    _vector_store_available = bool(bucket)
    return _vector_store_available


def _rag_search(query: str, n: int = 5) -> list[dict]:
    try:
        from vector_store import search
        return search(query, n_results=n)
    except Exception as e:
        return [{"text": f"Vector store unavailable: {e}", "source": "", "title": "", "distance": 1.0}]


def lookup_topic(topic: str) -> dict:
    """Retrieve information about a topic from the PDF knowledge base via semantic search."""
    if not _check_vector_store():
        return {
            "found": False,
            "topic": topic,
            "message": "Knowledge base not yet initialized. Run ingest.py and set KNOWLEDGE_BUCKET.",
        }

    hits = _rag_search(topic, n=5)
    if not hits or hits[0]["distance"] > 0.8:
        return {
            "found": False,
            "topic": topic,
            "message": f"No relevant content found for '{topic}' in the knowledge base.",
            "available_topics": AVAILABLE_TOPICS,
        }

    sources = list({h["title"] for h in hits if h["title"]})
    passages = [h["text"] for h in hits]

    return {
        "found": True,
        "topic": topic,
        "sources": sources,
        "passages": passages,
        "note": "Content retrieved from: Attention Is All You Need (Vaswani et al.) and Formal Algorithms for Transformers (Phuong & Hutter)",
    }


def generate_quiz(topic: str, num_questions: int = 3) -> dict:
    """Generate quiz questions on a topic, drawing from the paper-based quiz bank."""
    topic_lower = topic.lower().strip()

    matched_key = None
    for key in QUIZ_BANK:
        if key in topic_lower or topic_lower in key or any(w in topic_lower for w in key.split()):
            matched_key = key
            break

    if matched_key is None:
        all_questions = [q for questions in QUIZ_BANK.values() for q in questions]
        pool = random.sample(all_questions, min(num_questions, len(all_questions)))
        return {
            "found": True,
            "topic": topic,
            "note": f"No exact quiz for '{topic}', showing general transformer questions.",
            "num_questions": len(pool),
            "questions": [{"number": i+1, "question": q["q"], "answer": q["a"]} for i, q in enumerate(pool)],
        }

    pool = QUIZ_BANK[matched_key]
    selected = random.sample(pool, min(num_questions, len(pool)))
    return {
        "found": True,
        "topic": matched_key,
        "source": "Based on: Attention Is All You Need & Formal Algorithms for Transformers",
        "num_questions": len(selected),
        "questions": [{"number": i+1, "question": q["q"], "answer": q["a"]} for i, q in enumerate(selected)],
    }


def list_topics() -> dict:
    """List all topics covered in the knowledge base papers."""
    return {
        "available_topics": AVAILABLE_TOPICS,
        "count": len(AVAILABLE_TOPICS),
        "sources": [
            "Attention Is All You Need (Vaswani et al., 2017)",
            "Formal Algorithms for Transformers (Phuong & Hutter, DeepMind)",
        ],
        "description": "These topics are grounded in the two source papers. Ask me anything about transformers, attention, BERT, GPT, training, or formal algorithms.",
    }


def compare_topics(topic_a: str, topic_b: str) -> dict:
    """Compare two topics by retrieving relevant passages for each."""
    result_a = lookup_topic(topic_a)
    result_b = lookup_topic(topic_b)

    if not result_a["found"] and not result_b["found"]:
        return {"found": False, "message": f"Could not find content for '{topic_a}' or '{topic_b}'."}

    return {
        "found": True,
        "topic_a": {
            "name": topic_a,
            "passages": result_a.get("passages", [])[:2],
            "sources": result_a.get("sources", []),
        },
        "topic_b": {
            "name": topic_b,
            "passages": result_b.get("passages", [])[:2],
            "sources": result_b.get("sources", []),
        },
        "note": "Content from: Attention Is All You Need & Formal Algorithms for Transformers",
    }


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_topic",
            "description": (
                "Look up information about a transformer/LLM topic from the academic paper knowledge base. "
                "Uses semantic search over 'Attention Is All You Need' and 'Formal Algorithms for Transformers'. "
                "Use this for any question about attention, transformers, BERT, GPT, positional encoding, training, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or question to look up (e.g. 'multi-head attention', 'positional encoding', 'BERT training').",
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
            "description": "Generate quiz questions from the paper knowledge base to test understanding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to quiz on."},
                    "num_questions": {"type": "integer", "description": "Number of questions (default 3).", "default": 3},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_topics",
            "description": "List all topics covered in the knowledge base papers.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_topics",
            "description": "Compare two concepts side-by-side using retrieved passages from the papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_a": {"type": "string", "description": "First concept."},
                    "topic_b": {"type": "string", "description": "Second concept."},
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

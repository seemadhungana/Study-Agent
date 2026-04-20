# Technical Report — Study Assistant Agent

**Course:** Context Augmented AI (Spring 2026)  
**Student:** Seema Dhungana  
**Date:** April 2026  

---

## a. Problem and Use Case

**Problem:** Students studying AI/ML concepts need an on-demand tutor that gives accurate, structured explanations rather than hallucinated answers. Traditional chatbots make up facts; flashcard apps don't explain; search engines require sifting through results.

**User:** A student in an AI/ML course who wants to learn, review, or self-test on concepts like transformers, RAG, agents, and function calling.

**What the application does:**
- Explains topic concepts with summaries, key terms, and specific facts retrieved from a curated knowledge base
- Generates randomized quizzes for self-testing
- Compares two concepts side-by-side
- Maintains conversation history within a session so follow-up questions work naturally

---

## b. System Design

### High-Level Architecture

```
Browser (index.html)
    │  POST /chat  (JSON: message, session_id)
    ▼
FastAPI Server (main.py)
    │  session history lookup
    ▼
Agent Loop (agent.py)
    │  OpenAI Chat Completions API + tool definitions
    │  ┌─────────────────────────────────────┐
    │  │  LLM decides: call tool or respond  │
    │  └─────────────────────────────────────┘
    │  Tool dispatch (tools.py)
    │  ┌──────────────┬────────────────┬──────────────┬────────────────┐
    │  │ lookup_topic │ generate_quiz  │ list_topics  │ compare_topics │
    │  └──────────────┴────────────────┴──────────────┴────────────────┘
    │  Tool results fed back into context → final LLM response
    ▼
Observability (observability.py)
    │  JSONL log per event (request_start, tool_call, response_complete)
    │  In-memory metrics aggregation
    ▼
Response JSON → Browser
```

### Main Components

| Component | File | Role |
|---|---|---|
| Web server | `main.py` | FastAPI app, session store, `/chat`, `/metrics`, `/traces` endpoints |
| Agent loop | `agent.py` | Multi-step LLM loop with tool dispatch and history management |
| Tools | `tools.py` | Knowledge base, quiz bank, tool definitions (JSON Schema), dispatch registry |
| Observability | `observability.py` | Structured JSONL logging, in-memory metrics computation |
| Frontend | `static/index.html` | Chat UI with topic sidebar, metrics panel, conversation display |

### Agentic Behavior Implementation

Each turn, `agent.py` calls the OpenAI Chat Completions API with `tools=TOOL_DEFINITIONS` and `tool_choice="auto"`. The model either:
1. Returns a `tool_calls` list → the app dispatches each call, appends tool results to messages, and loops
2. Returns a plain `content` message → the loop exits and returns the response

The loop runs up to `max_iterations=5` to support multi-step reasoning (e.g., the LLM could look up two topics before comparing them). This is the core decision point: the LLM — not the application — controls whether and which tool is called.

---

## c. Why the System is Agentic

**Decisions the LLM makes each turn:**
- Whether to call any tool at all (simple greetings get direct responses; topic questions trigger `lookup_topic`)
- Which specific tool is appropriate — a "quiz me" request triggers `generate_quiz`, not `lookup_topic`
- What arguments to pass (it extracts the topic name from natural language)
- Whether to chain tool calls (a compare request may call `lookup_topic` twice or `compare_topics` once)

**This is meaningfully agentic because:**
- The same user input can result in different tool paths depending on phrasing and context
- The model can choose *not* to call a tool when it's not needed
- The workflow is not fixed — the LLM determines the sequence of tool calls dynamically

**This is NOT a fixed pipeline** — there is no hardcoded "always call lookup_topic first." The model makes the routing decision.

---

## d. Technical Choices and Rationale

| Choice | Rationale |
|---|---|
| **GPT-4o-mini** | Strong function calling support, low latency, cost-effective for a study app |
| **OpenAI function calling** | Native support for structured tool invocation; no external orchestration framework needed |
| **FastAPI** | Async, fast, clean Pydantic validation, easy to deploy on Render/Railway |
| **In-memory session store** | Sufficient for a demo/single-instance deployment; easy to swap for Redis later |
| **Custom knowledge base (dict)** | Guarantees factual accuracy for course-specific content; avoids hallucination risk of pure LLM answers |
| **JSONL file logging** | Zero-dependency observability; human-readable; easily piped into any log aggregator |
| **No RAG** | The knowledge base is small and curated; vector search would add complexity without meaningful benefit at this scale |
| **Vanilla HTML/JS frontend** | No build step, no framework overhead; runs on any static host; easy to inspect |

---

## e. Observability

**Mechanism:** Custom structured logging to `app/logs/agent_traces.jsonl` + in-memory metrics aggregation.

**What it captures:**

| Event Type | Fields |
|---|---|
| `request_start` | trace_id, session_id, timestamp, user_message |
| `tool_call` | trace_id, tool_name, arguments, result_preview, success |
| `response_complete` | trace_id, session_id, latency_ms, tool_calls_made, response_preview, error |

**How it helps inspect system behavior:**
- `trace_id` links all events for one request — easy to trace a failure end-to-end
- `tool_call.success=false` immediately flags retrieval failures
- `response_complete.error` captures exceptions with context
- `latency_ms` shows which requests were slow
- `result_preview` lets me verify the tool returned reasonable content without logging full payloads

The `/metrics` endpoint aggregates these in real-time; the `/traces` endpoint returns the last N raw records. Both are displayed in the frontend's sidebar panel.

---

## f. Metrics

### Metric 1: Tool Call Success Rate (Quality Metric)

**Definition:** `successful_tool_calls / total_tool_calls × 100`

**Why it matters:** If the LLM is calling tools with invalid topic names (e.g., asking for a topic not in the knowledge base), the tool returns `found: false`. A low success rate means the LLM is making poor tool decisions — either the system prompt needs refinement or the knowledge base needs to expand.

**How tracked:** `observability.py` increments `successful_tool_calls` when the tool result contains `found: true` and no `error` key.

### Metric 2: Average Response Latency (Operational Metric)

**Definition:** `total_latency_ms / total_requests` (wall-clock time from request receipt to response sent)

**Why it matters:** If latency spikes above ~3 seconds, the experience degrades for students. High latency could indicate API rate limiting, a long agentic chain, or a slow model response. This helps decide whether to switch models or add caching.

**How tracked:** `time.perf_counter()` wraps each `/chat` call; accumulated in `_metrics["total_latency_ms"]`.

---

## g. Evaluation

### Test Scenarios

| Scenario | Input | Expected | Observed |
|---|---|---|---|
| Topic lookup | "Explain transformers" | Calls `lookup_topic("transformers")`, returns summary + facts | ✅ Correct tool called, factual answer |
| Quiz request | "Quiz me on RAG with 4 questions" | Calls `generate_quiz("rag", 4)`, returns 4 Q&A pairs | ✅ Correct count, relevant questions |
| Comparison | "Compare agents and RAG" | Calls `compare_topics("agents", "RAG")`, returns side-by-side | ✅ Both topics retrieved correctly |
| Topic listing | "What can you help me with?" | Calls `list_topics()`, returns all 7 topics | ✅ Tool correctly selected |
| Out-of-scope | "What's the weather today?" | No tool call, direct response | ✅ Model correctly skips tools |
| Ambiguous topic | "Tell me about attention" | Calls `lookup_topic("attention")` → no match | ⚠️ Fails silently; LLM improvises |
| Follow-up | "Give me more facts about that" | Uses conversation history correctly | ✅ Context preserved across turns |

### Successes

- Tool selection is accurate for clear, in-scope requests
- Knowledge base entries are authoritative and factually grounded
- Conversation history keeps multi-turn sessions coherent
- The model never invents facts when a tool is available — it defers to the tool result

### Failure Cases and Struggles

- **Synonym mismatch:** User says "attention mechanism" → `lookup_topic` misses it (it's under "transformers"). The model then fills in from its own weights, which may or may not be accurate.
- **Unknown topics:** Requests about topics outside the knowledge base (e.g., "reinforcement learning") get no tool support — the LLM answers from training data, increasing hallucination risk.
- **Multi-intent requests:** "Explain RAG and quiz me on it" can work but sometimes produces two separate tool calls in one turn or combines them; behavior is inconsistent.

### Tradeoffs

| Tradeoff | Decision |
|---|---|
| Speed vs. accuracy | Using a curated KB instead of RAG is faster but limits scope |
| Simplicity vs. flexibility | In-memory session store is simple but lost on server restart |
| Strict tool definitions vs. generality | Narrow JSON schemas improve precision but miss paraphrased requests |

### What I Would Improve

1. **Expand knowledge base** to more topics (RL, diffusion models, fine-tuning) — this is the single biggest quality lever
2. **Add fuzzy topic matching** (e.g., embeddings-based lookup) so "attention mechanism" correctly maps to "transformers"
3. **Persistent sessions** via a database (SQLite or Redis) so conversations survive restarts
4. **Formal evaluation set** — 20+ representative prompts with ground-truth expected tool calls; compute precision/recall on tool selection
5. **LangSmith integration** for richer trace visualization than the custom JSONL approach

---

## h. Deployment

**Platform:** Render (render.com) — a free-tier PaaS that supports Python web services.

**How it's deployed:**
- The GitHub repo is connected to Render as a Web Service
- Build command: `pip install -r requirements.txt`
- Start command: `cd app && uvicorn main:app --host 0.0.0.0 --port $PORT`
- `OPENAI_API_KEY` is set as a secret environment variable in Render's dashboard (never in the repo)

**Practical constraints:**
- Render free tier spins down after 15 minutes of inactivity — first request after sleep has ~30s cold start
- No persistent disk on free tier → JSONL log file is lost on restart; metrics reset to zero
- For a production deployment, JSONL logs would be written to S3 or a managed log service, and sessions would be stored in Redis

**Public URL:** `https://study-assistant-agent.onrender.com` *(update before submission)*

---

## i. Reflection

### What I Learned

- **Function calling is the cleanest way to build tool-using agents** — no parsing, no regex, just structured JSON. The schema design (description, parameter names) matters enormously for whether the LLM picks the right tool.
- **Observability must be designed in from the start** — retrofitting logging is painful. Having `trace_id` thread through all events made debugging much easier.
- **Curated knowledge beats RAG at small scale** — for a fixed topic set, a Python dict with known-good facts outperforms a vector store in accuracy, latency, and simplicity.
- **System prompt wording directly affects tool selection** — small phrasing changes ("ALWAYS call lookup_topic first") measurably changed which tool was picked.

### What I Would Improve with More Time

1. A persistent vector store (e.g., Chroma or pgvector) to cover topics beyond the hardcoded knowledge base
2. An evaluation harness that scores tool call accuracy and answer quality automatically
3. A proper metrics dashboard (Grafana or Streamlit) rather than the raw JSON endpoint
4. User authentication so multiple students can have private session histories
5. Streaming responses (SSE or WebSocket) to show the agent's thinking in real time

### Design Choices I Would Revisit

- **In-memory sessions** — I would use a database from day one, even SQLite
- **Single knowledge base file** — a YAML or JSON file per topic would be easier to maintain and extend than a Python dict
- **No re-ranking** — for a real RAG upgrade, I would include a cross-encoder re-ranker to improve retrieval precision before grounding the LLM

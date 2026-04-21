# Technical Report — Study Assistant Agent

**Course:** Context Augmented AI (Spring 2026)
**Student:** Seema Dhungana
**Date:** April 2026

---

## a. Problem and Use Case

**Problem:** Students studying transformer architectures and large language models need an on-demand tutor grounded in actual academic papers rather than generic internet knowledge. Traditional chatbots hallucinate details; search engines require sifting through dense PDFs; flashcard apps don't explain.

**User:** A student in an AI/ML course who wants to learn, review, or self-test on concepts from foundational transformer research papers.

**What the application does:**
- Answers questions about transformer concepts by retrieving relevant passages directly from two academic papers via semantic search
- Generates quizzes drawn from paper-based question banks for self-testing
- Compares two concepts side-by-side using retrieved evidence
- Maintains conversation history within a session so follow-up questions work naturally
- Falls back gracefully to the LLM's own knowledge for out-of-scope topics, clearly acknowledging the limitation

---

## b. System Design

### High-Level Architecture

```
Browser (index.html)
    │  POST /chat  (JSON: message, session_id)
    ▼
FastAPI Server (main.py)  [AWS Lambda via Mangum]
    │  session history lookup
    ▼
DynamoDB  ←──────────────────────────────────────────┐
    │  conversation history (PK=SESSION#id, SK=HISTORY)│
    ▼                                                  │
Agent Loop (agent.py)                                  │
    │  Amazon Bedrock Converse API (Claude 3 Haiku)     │
    │  ┌─────────────────────────────────────┐          │
    │  │  LLM decides: call tool or respond  │          │
    │  └─────────────────────────────────────┘          │
    │  Tool dispatch (tools.py)                         │
    │  ┌──────────────┬───────────────┬─────────────┬──────────────┐
    │  │ lookup_topic │ generate_quiz │ list_topics │ compare_topics│
    │  └──────────────┴───────────────┴─────────────┴──────────────┘
    │         │
    │         ▼
    │   Vector Store (vector_store.py)
    │   Bedrock Titan Embeddings → cosine similarity search
    │   ChromaDB index stored as gzipped JSON in S3
    │   (downloaded to /tmp on Lambda cold start)
    │
    ▼
Observability (observability.py)
    │  JSONL trace log (/tmp/agent_traces.jsonl)
    │  DynamoDB metric counters (PK=METRICS, SK=GLOBAL)
    ▼
Response JSON → Browser
```

### Main Components

| Component | File | Role |
|---|---|---|
| Web server | `main.py` | FastAPI app + Mangum Lambda adapter; session management; `/chat`, `/metrics`, `/traces` endpoints |
| Agent loop | `agent.py` | Multi-step Bedrock Converse API loop with tool dispatch; handles `end_turn` vs `tool_use` stop reasons |
| Tools | `tools.py` | 4 tool functions; paper-based quiz bank; TOOL_DEFINITIONS in Bedrock toolSpec format |
| Vector store | `vector_store.py` | Downloads embeddings from S3 on cold start; cosine similarity search via numpy |
| Ingestion | `ingest.py` | One-time script: extracts PDF text, chunks, embeds via Bedrock Titan, uploads to S3 |
| Session store | `session_store.py` | DynamoDB-backed conversation history with 7-day TTL and 20-message cap |
| Observability | `observability.py` | JSONL trace logging + DynamoDB atomic counters for persistent metrics |
| Frontend | `static/index.html` | Dark-mode chat UI with topic sidebar, live metrics panel, typing indicator |
| Infrastructure | `template.yaml` | AWS SAM: Lambda, HTTP API Gateway, DynamoDB, S3 bucket, IAM policies |

### Agentic Behavior Implementation

Each turn, `agent.py` calls the Bedrock Converse API with `toolConfig` containing 4 tool definitions. The model returns either:
1. `stopReason: "tool_use"` → the app dispatches each tool call, appends results as a `user` turn, and loops
2. `stopReason: "end_turn"` → the loop exits and returns the text response

The loop runs up to `max_iterations=5`. The LLM — not the application — decides whether to call a tool, which tool, and with what arguments on every turn.

---

## c. Why the System is Agentic

**Decisions the LLM makes each turn:**
- Whether to call any tool at all (greetings → direct response; topic questions → `lookup_topic`)
- Which specific tool is appropriate ("quiz me" → `generate_quiz`; "what's available" → `list_topics`)
- What arguments to pass (extracts topic names from natural language)
- Whether to chain calls (a compare request may invoke `lookup_topic` twice before synthesizing)

**This is meaningfully agentic because:**
- The same user input can result in different tool paths depending on phrasing and context
- The model can choose *not* to call a tool when the topic is out of scope
- The workflow is not fixed — the LLM dynamically determines the sequence of operations
- For out-of-scope queries (e.g., "What is LoRA?"), the LLM correctly skips tool use and answers directly, acknowledging the knowledge base limitation

**This is NOT a fixed pipeline** — there is no hardcoded "always call lookup_topic first." The routing decision belongs entirely to the model.

---

## d. Technical Choices and Rationale

| Choice | Rationale |
|---|---|
| **Amazon Bedrock (Claude 3 Haiku)** | Native AWS integration — no separate API key; IAM role-based auth; strong tool-use support; cost-effective at ~$0.25/1M input tokens |
| **Bedrock Converse API** | Unified interface across Claude models; native `tool_use` / `end_turn` stop reasons make agentic loops clean |
| **Bedrock Titan Embeddings v2** | AWS-native embedding model; no external API; used for both ingestion and query embedding |
| **Cosine similarity + numpy** | Pure Python vector search — no native binaries, fits within Lambda's 250MB unzipped limit (ChromaDB was too large) |
| **S3 for embedding storage** | Persistent, cheap (~$0.01/month for 38 chunks), accessible from Lambda; embeddings downloaded once per cold start |
| **DynamoDB for sessions + metrics** | Already provisioned; atomic `ADD` expressions make metric counters race-condition-safe across concurrent Lambda invocations |
| **FastAPI + Mangum** | Minimal adapter to run ASGI on Lambda; same code runs locally with `uvicorn` |
| **AWS SAM** | Infrastructure-as-code for Lambda, API Gateway, DynamoDB, S3; reproducible one-command deployment |
| **PDF chunking (400 words, 50 overlap)** | Balances semantic coherence with embedding context limits; overlap reduces information loss at chunk boundaries |
| **Two source papers** | "Attention Is All You Need" (Vaswani et al., 2017) and "Formal Algorithms for Transformers" (Phuong & Hutter, DeepMind) — canonical, academically rigorous sources for transformer content |

---

## e. Observability

**Mechanism:** Two-layer observability — ephemeral JSONL traces in `/tmp` for per-request inspection, and persistent DynamoDB atomic counters for cumulative metrics.

**What it captures:**

| Event Type | Fields |
|---|---|
| `request_start` | trace_id, session_id, timestamp, user_message |
| `tool_call` | trace_id, tool_name, arguments, result_preview (300 chars), success |
| `response_complete` | trace_id, session_id, latency_ms, tool_calls_made, response_preview, error |

**DynamoDB metric counters (PK=METRICS, SK=GLOBAL):**
- `total_requests`, `total_tool_calls`, `successful_tool_calls`, `total_latency_ms`, `errors`
- Per-tool counters: `tool_lookup_topic_calls`, `tool_generate_quiz_calls`, etc.

**How it helps:**
- `trace_id` links all events for one request end-to-end
- `tool_call.success=false` immediately flags retrieval failures vs. successful retrievals
- `latency_ms` per request identifies slow agentic chains (multiple tool calls compound latency)
- DynamoDB counters persist across Lambda cold starts — metrics accumulate across all invocations
- `/metrics` endpoint serves aggregated stats to the live frontend panel in real time
- `/traces` endpoint returns the last N raw JSONL records for debugging

---

## f. Metrics

### Metric 1: Tool Call Success Rate (Quality Metric)

**Definition:** `successful_tool_calls / total_tool_calls × 100`

**A tool call is successful if** the result JSON contains no `"error"` key and does not have `"found": false`.

**Why it matters:** If the LLM is calling tools but they return no results (topic not found, retrieval miss), success rate drops. A low rate signals that either the system prompt needs refinement, the retrieval is missing relevant chunks, or the LLM is calling tools for out-of-scope queries that shouldn't trigger tool use at all.

**How tracked:** `agent.py` increments `successful_tool_calls` per successful dispatch; `log_response_complete()` writes both counts to DynamoDB atomically.

### Metric 2: Average Response Latency (Operational Metric)

**Definition:** `total_latency_ms / total_requests` — wall-clock time from request receipt to response sent

**Why it matters:** Each tool call adds a round-trip: one Bedrock Converse call to decide, one embedding call, one S3/numpy search, another Converse call to synthesize. Multi-tool turns compound this. If latency exceeds ~5 seconds the study experience degrades. This metric helps identify whether slow responses are from the model, the retrieval, or the agentic chain length.

**Observed range:** 4,000–6,500ms per request (single tool call turns). Breakdown: ~500ms Titan embedding, ~100ms S3 download (warm), ~3,500ms Claude synthesis.

**How tracked:** `time.perf_counter()` wraps each `/chat` handler; accumulated in DynamoDB via `ADD total_latency_ms`.

---

## g. Evaluation

### Test Scenarios

Evaluation was conducted across two categories: in-scope queries (topics covered in the two papers) and out-of-scope queries (topics outside the knowledge base). Six representative scenarios were tested.

#### In-Scope Queries (Expected: tool called, relevant content retrieved)

| Query | Tool Called | Result | Assessment |
|---|---|---|---|
| "Explain scaled dot-product attention" | `lookup_topic` | Retrieved relevant attention formula passages; coherent explanation | ✅ Success |
| "Quiz me on transformer architecture with 4 questions" | `generate_quiz` | Returned 4 paper-grounded Q&A pairs correctly | ✅ Success |
| "Compare BERT and GPT architectures" | `compare_topics` | Retrieved passages for both; synthesized accurate contrast | ✅ Success |

#### Stress Tests — Specific Facts from the Papers

| Query | Tool Called | Result | Assessment |
|---|---|---|---|
| "What is the exact BLEU score the transformer achieved on English-German translation?" | `lookup_topic` | Tool called; chunks retrieved did not contain the results table. LLM stated it "couldn't confirm" the score from the knowledge base — then stated 28.4 BLEU anyway from its own training data | ⚠️ Retrieval miss + over-hedging |
| "Explain Algorithm 1 from the Formal Algorithms paper" | None | LLM answered directly from training data without calling a tool; provided a plausible but unverified reconstruction of the algorithm | ⚠️ Tool not triggered; no grounding |
| "What dropout rate did the original transformer use?" | `lookup_topic` | Tool called; retrieved chunks did not contain the hyperparameter. LLM hedged, said it couldn't confirm, then answered 0.1 from its own knowledge | ⚠️ Retrieval miss + over-hedging |

#### Out-of-Scope Queries (Expected: no tool call, graceful fallback)

| Query | Tool Called | Result | Assessment |
|---|---|---|---|
| "What is RAG and how does it work?" | None | LLM answered from training data, noted the topic wasn't in the knowledge base | ✅ Correct fallback |
| "How do I fine-tune a model with LoRA?" | None | LLM answered directly, acknowledged knowledge base limitation | ✅ Correct fallback |
| "What is the context window of Claude?" | None | LLM answered from training data without tool use | ✅ Correct fallback |

### Successes

- Tool selection is accurate for clearly in-scope, natural-language requests
- Out-of-scope queries are handled gracefully — the LLM does not force tool calls where none are appropriate and acknowledges the knowledge base boundary to the user
- Conversation history is preserved across turns; follow-up questions ("give me more detail on that") work correctly
- Metrics accumulate persistently across Lambda cold starts via DynamoDB

### Failure Cases and Root Causes

**1. Retrieval misses on specific numeric facts**
Specific numbers (BLEU scores, dropout rates, layer counts) appear in dense results tables or brief clauses within longer paragraphs. At 400-word chunk granularity, these facts are typically embedded in a chunk dominated by surrounding context, causing the embedding to rank other chunks higher. The specific fact is not in the top-5 results.

**2. LLM over-hedging after retrieval miss**
When the retrieved chunks don't contain the answer, the LLM incorrectly defers entirely to the knowledge base and says it "cannot confirm" — even when it knows the answer from its own training. This produces a confusing response where the model states it can't answer, then answers anyway. This is a system prompt design issue: the prompt should instruct the model to supplement missing retrieval results with its own knowledge more confidently.

**3. Tool not triggered for highly specific algorithmic queries**
"Explain Algorithm 1 from the Formal Algorithms paper" did not trigger a tool call. The LLM interpreted this as a general knowledge question rather than a retrieval task. A more explicit system prompt instruction — "if the user references a specific section or algorithm from either paper, always call lookup_topic" — would fix this.

### Tradeoffs

| Tradeoff | Decision Made | Consequence |
|---|---|---|
| Chunk size (400 words) vs. specificity | Larger chunks for semantic coherence | Specific numeric facts get diluted in embedding space |
| numpy cosine search vs. ChromaDB | numpy to stay within Lambda size limit | No HNSW indexing; O(n) search over 38 chunks (negligible at this scale) |
| Bedrock Titan embeddings vs. OpenAI | Titan for AWS-native auth | Slightly lower embedding quality than OpenAI ada-002 |
| Graceful fallback vs. strict grounding | Allow fallback to LLM knowledge | Increases risk of ungrounded answers for out-of-scope queries |

### What I Would Improve

1. **Smaller chunk size (100–150 words)** with sentence-level splitting to preserve specific numeric facts in their own chunks
2. **System prompt refinement** — instruct the LLM to supplement retrieval results with its own knowledge confidently rather than over-hedging
3. **Metadata filtering** — tag chunks by section (Introduction, Results, Algorithm) so queries about specific sections can filter before embedding search
4. **Re-ranking** — add a cross-encoder pass over the top-10 retrieved chunks to improve precision before grounding the LLM
5. **Formal evaluation set** — 20+ prompts with ground-truth expected tool calls and expected answer content; compute retrieval precision@k and tool selection accuracy

---

## h. Deployment

**Platform:** AWS Lambda + Amazon API Gateway (HTTP API v2), deployed via AWS SAM.

**Infrastructure (defined in `template.yaml`):**
- **Lambda function:** Python 3.12, 1024MB memory, 30s timeout; Mangum adapter wraps FastAPI
- **API Gateway:** HTTP API with CORS enabled; routes all paths to Lambda
- **DynamoDB:** `StudyAssistantSessions` table; PAY_PER_REQUEST billing; 7-day TTL on session items; also stores persistent metrics under `PK=METRICS`
- **S3 bucket:** `study-assistant-knowledge-991452971884`; stores `knowledge/embeddings.json.gz` (38 chunks, ~2MB)
- **IAM:** Lambda execution role with `DynamoDBCrudPolicy`, `S3ReadPolicy`, and `bedrock:InvokeModel` on all Bedrock resources

**Deployment process:**
```bash
python ingest.py          # one-time: embed PDFs, upload to S3
sam build                 # package Lambda with dependencies
sam deploy                # deploy/update CloudFormation stack
```

**Practical constraints:**
- Lambda `/tmp` (512MB) is ephemeral — JSONL traces reset on cold start; metrics are in DynamoDB so they persist
- Cold start latency (~2–3s extra) occurs when Lambda container is recycled; first request after idle period is slower
- Bedrock Titan embedding adds ~500ms per query for the semantic search step
- S3 embeddings download (~2MB) only happens on cold start; warm invocations reuse the cached numpy arrays in memory

**Public URL:** https://kpu7oyg4pc.execute-api.us-east-1.amazonaws.com

---

## i. Reflection

### What I Learned

- **Infrastructure-as-code pays off immediately** — having everything in `template.yaml` meant I could tear down and redeploy the entire stack in minutes when things broke. Debugging IAM permission errors would have been much harder without SAM's clear error messages.
- **Lambda size limits are a real constraint** — ChromaDB's native binaries exceeded Lambda's 250MB unzipped limit, requiring a full redesign to pure-Python numpy cosine search. Choosing dependencies for serverless requires checking unzipped size, not just install size.
- **Chunk granularity directly determines retrieval quality** — 400-word chunks were too coarse for specific fact retrieval. The evaluation stress tests revealed this clearly: broad conceptual questions worked well, but precise numeric lookups failed.
- **System prompt wording determines tool selection behavior** — small changes to the system prompt ("ALWAYS call lookup_topic first" vs. "use tools when relevant") had measurable effects on whether the LLM triggered tool calls for borderline queries.
- **DynamoDB atomic counters are the right tool for serverless metrics** — in-memory counters reset on cold start; file-based logs reset when `/tmp` is recycled. DynamoDB `ADD` expressions are atomic and persist indefinitely at negligible cost.
- **AWS IAM is the biggest deployment friction** — getting the right policies attached to the right role took more debugging time than writing the application code.

### What I Would Improve with More Time

1. **Finer chunk granularity** (100–150 words, sentence-aware splitting) to improve specific fact retrieval
2. **Re-ranking layer** — a cross-encoder pass over top-10 retrieved chunks before grounding the LLM
3. **Streaming responses** via Server-Sent Events so the user sees the agent's answer token-by-token rather than waiting 5+ seconds
4. **Expand the knowledge base** to additional papers (BERT, GPT-2, Flash Attention) so the agent covers more of the course curriculum
5. **Formal evaluation harness** — automated test suite with ground-truth tool call expectations and answer quality scoring

### Design Choices I Would Revisit

- **400-word chunk size** — the stress tests showed this is too coarse; I would use 100–150 word chunks with sentence-boundary awareness from the start
- **Single embedding model** — Bedrock Titan v2 is convenient but I would benchmark against Cohere or OpenAI embeddings for this domain to verify retrieval quality
- **No streaming** — the 4–6 second wait per response is the biggest UX friction point; I would implement SSE streaming from the start rather than retrofitting it

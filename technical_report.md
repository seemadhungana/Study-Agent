# Technical Report — Study Assistant Agent

**Course:** Context Augmented AI (Spring 2026)
**Student:** Seema Dhungana
**Date:** April 21 2026

---

## a. Problem and Use Case

Students studying transformer architectures and large language models often struggle to find reliable, on-demand explanations grounded in actual academic papers. Traditional chatbots tend to hallucinate details, search engines require manually sifting through dense PDFs, and flashcard apps don't explain concepts. I built a Study Assistant Agent to fill this gap. The tutor answers questions, generates quizzes, and compares concepts by retrieving evidence directly from foundational research papers rather than relying on generic internet knowledge.

The primary user is a student in an AI/ML course who wants to learn, review, or self-test on transformer concepts from academic sources. The application answers questions about transformer topics by semantically searching two academic papers, generates paper-grounded quizzes for self-testing, compares two concepts side-by-side using retrieved evidence, and maintains conversation history within a session so follow-up questions work naturally. When a topic falls outside the knowledge base, the agent gracefully falls back to the model's own knowledge while acknowledging the limitation.

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

Each turn, `agent.py` calls the Bedrock Converse API with a `toolConfig` containing four tool definitions. The model returns either a `stopReason: "tool_use"`, at which point the app dispatches each tool call, appends the results as a new user turn, and loops, or a `stopReason: "end_turn"`, which exits the loop and returns the text response. The loop runs up to `max_iterations=5`. Critically, the LLM, not the application, decides on every turn whether to call a tool, which tool to call, and what arguments to pass.

---

## c. Why the System is Agentic

The system is meaningfully agentic because the LLM makes real routing decisions on every turn rather than following a fixed pipeline. Depending on the user's phrasing and conversation context, the same input can produce different tool paths: a greeting gets a direct response, a topic question triggers `lookup_topic`, "quiz me" triggers `generate_quiz`, and "what's available" triggers `list_topics`. The model also decides what arguments to pass by extracting topic names from natural language, and can chain calls, a compare request may invoke `lookup_topic` twice before synthesizing an answer.

Equally important, the model knows when *not* to call a tool. For out-of-scope queries like "What is LoRA?", the LLM correctly skips tool use and answers directly from its training data, acknowledging the knowledge base limitation. There is no hardcoded "always call `lookup_topic` first" logic anywhere in the application, the routing decision belongs entirely to the model.

---

## d. Technical Choices and Rationale

The core model is **Amazon Bedrock with Claude 3 Haiku**, chosen for its native AWS integration (no separate API key, IAM role-based auth), strong tool-use support, and cost-effectiveness. I used the **Bedrock Converse API** specifically because its unified interface across Claude models makes the agentic loop clean: the `tool_use` and `end_turn` stop reasons map directly onto the agent's branch logic.

For embeddings, I used **Bedrock Titan Embeddings v2**, an AWS-native model that requires no external API and is used consistently for both ingestion and query embedding. Vector search is implemented with **cosine similarity over numpy arrays** rather than ChromaDB, because ChromaDB's native binaries exceeded Lambda's 250MB unzipped size limit, this was a hard constraint, not a preference. The embedding index is stored as a gzipped JSON file in **S3**, downloaded once per cold start and cached in memory for warm invocations.

**DynamoDB** handles both session history and persistent metrics. Atomic `ADD` expressions make metric counters race-condition-safe across concurrent Lambda invocations, which matters in serverless environments where multiple containers can run simultaneously. The web server uses **FastAPI + Mangum**, a minimal ASGI adapter that lets the same code run locally with `uvicorn` and on Lambda without modification. Infrastructure is managed via **AWS SAM**, enabling reproducible one-command deployments. PDF chunking uses **400-word chunks with 50-word overlap**, balancing semantic coherence with embedding context limits; the overlap reduces information loss at chunk boundaries. The two source papers,  "Attention Is All You Need" (Vaswani et al., 2017) and "Formal Algorithms for Transformers" (Phuong & Hutter, DeepMind), were chosen as the canonical, academically rigorous foundations for transformer content.

---

## e. Observability

The system uses a two-layer observability approach: ephemeral JSONL traces written to `/tmp` for per-request inspection, and persistent DynamoDB atomic counters for cumulative metrics that survive Lambda cold starts.

**What it captures:**

| Event Type | Fields |
|---|---|
| `request_start` | trace_id, session_id, timestamp, user_message |
| `tool_call` | trace_id, tool_name, arguments, result_preview (300 chars), success |
| `response_complete` | trace_id, session_id, latency_ms, tool_calls_made, response_preview, error |

**DynamoDB metric counters (PK=METRICS, SK=GLOBAL):**
- `total_requests`, `total_tool_calls`, `successful_tool_calls`, `total_latency_ms`, `errors`
- Per-tool counters: `tool_lookup_topic_calls`, `tool_generate_quiz_calls`, etc.

The `trace_id` links all events for a single request end-to-end, making it possible to reconstruct exactly what the agent did for any given query. A `tool_call.success=false` flag immediately distinguishes retrieval failures from successful ones, and `latency_ms` per request helps identify slow agentic chains where multiple tool calls compound response time. The `/metrics` endpoint serves aggregated stats to the live frontend panel, and `/traces` returns the last N raw JSONL records for debugging.

---

## f. Metrics

### Metric 1: Tool Call Success Rate (Quality Metric)

Tool call success rate is defined as `successful_tool_calls / total_tool_calls × 100`. A tool call is considered successful if the result JSON contains no `"error"` key and does not have `"found": false`. This metric matters because if the LLM is calling tools that return no useful results, due to a topic not found in the index or a retrieval miss, success rate drops and signals a problem. A low rate points to either a system prompt that needs refinement, retrieval that is missing relevant chunks, or the LLM calling tools for out-of-scope queries that shouldn't trigger tool use at all. Successful calls are incremented in `agent.py` per dispatch, and both counts are written to DynamoDB atomically via `log_response_complete()`.

### Metric 2: Average Response Latency (Operational Metric)

Average response latency is defined as `total_latency_ms / total_requests`, measured as wall-clock time from request receipt to response sent. This metric matters because each tool call adds a round-trip (one Bedrock Converse call to decide, one Titan embedding call, one S3/numpy search, and another Converse call to synthesize) and multi-tool turns compound this. If latency exceeds ~5 seconds the study experience degrades noticeably. Observed latency ranged from 4,000–6,500ms per request for single tool call turns, with the breakdown approximately: ~500ms Titan embedding, ~100ms S3 download (warm), ~3,500ms Claude synthesis. Latency is tracked via `time.perf_counter()` wrapping each `/chat` handler and accumulated in DynamoDB via `ADD total_latency_ms`.

---

## g. Evaluation

### Test Scenarios

Evaluation was conducted across two categories: in-scope queries (topics covered in the two papers) and out-of-scope queries (topics outside the knowledge base). Six representative scenarios were tested.

#### In-Scope Queries (Expected: tool called, relevant content retrieved)

| Query | Tool Called | Result | Assessment |
|---|---|---|---|
| "Explain scaled dot-product attention" | `lookup_topic` | Retrieved relevant attention formula passages; coherent explanation | Success |
| "Quiz me on transformer architecture with 4 questions" | `generate_quiz` | Returned 4 paper-grounded Q&A pairs correctly | Success |
| "Compare BERT and GPT architectures" | `compare_topics` | Retrieved passages for both; synthesized accurate contrast | Success |

#### Stress Tests — Specific Facts from the Papers

| Query | Tool Called | Result | Assessment |
|---|---|---|---|
| "What is the exact BLEU score the transformer achieved on English-German translation?" | `lookup_topic` | Tool called; chunks retrieved did not contain the results table. LLM stated it "couldn't confirm" the score from the knowledge base, then stated 28.4 BLEU anyway from its own training data | Retrieval miss + over-hedging |
| "Explain Algorithm 1 from the Formal Algorithms paper" | None | LLM answered directly from training data without calling a tool; provided a plausible but unverified reconstruction of the algorithm | Tool not triggered; no grounding |
| "What dropout rate did the original transformer use?" | `lookup_topic` | Tool called; retrieved chunks did not contain the hyperparameter. LLM hedged, said it couldn't confirm, then answered 0.1 from its own knowledge | Retrieval miss + over-hedging |

#### Out-of-Scope Queries (Expected: no tool call, graceful fallback)

| Query | Tool Called | Result | Assessment |
|---|---|---|---|
| "What is RAG and how does it work?" | None | LLM answered from training data, noted the topic wasn't in the knowledge base | Correct fallback |
| "How do I fine-tune a model with LoRA?" | None | LLM answered directly, acknowledged knowledge base limitation | Correct fallback |
| "What is the context window of Claude?" | None | LLM answered from training data without tool use | Correct fallback |

### Successes

The system handled clearly in-scope, natural-language requests well, tool selection was accurate and the retrieved content grounded the responses appropriately. Out-of-scope queries were handled gracefully: the LLM did not force tool calls where none were appropriate and correctly acknowledged the knowledge base boundary. Conversation history was preserved across turns, so follow-up questions like "give me more detail on that" worked correctly. Metrics also accumulated persistently across Lambda cold starts via DynamoDB, as expected.

### Failure Cases and Root Causes

The most significant failure was **retrieval misses on specific numeric facts**. Numbers like BLEU scores, dropout rates, and layer counts tend to appear in dense results tables or brief clauses within longer paragraphs. At 400-word chunk granularity, these facts are typically embedded inside chunks dominated by surrounding context, which causes the embedding to rank other chunks higher: the specific fact simply doesn't surface in the top-5 results.

A related failure was **LLM over-hedging after a retrieval miss**. When retrieved chunks don't contain the answer, the model incorrectly defers entirely to the knowledge base and says it "cannot confirm," even when it actually knows the answer from its own training. This produces a confusing response where the model says it can't answer and then answers anyway. This is a system prompt design issue: the prompt should instruct the model to supplement missing retrieval results with its own knowledge more confidently.

Finally, **tool not triggering for highly specific algorithmic queries** was an issue. "Explain Algorithm 1 from the Formal Algorithms paper" did not trigger a tool call because the LLM interpreted it as a general knowledge question. A more explicit system prompt instruction ("if the user references a specific section or algorithm from either paper, always call lookup_topic") would fix this.

### Tradeoffs

| Tradeoff | Decision Made | Consequence |
|---|---|---|
| Chunk size (400 words) vs. specificity | Larger chunks for semantic coherence | Specific numeric facts get diluted in embedding space |
| numpy cosine search vs. ChromaDB | numpy to stay within Lambda size limit | No HNSW indexing; O(n) search over 38 chunks (negligible at this scale) |
| Bedrock Titan embeddings vs. OpenAI | Titan for AWS-native auth | Slightly lower embedding quality than OpenAI ada-002 |
| Graceful fallback vs. strict grounding | Allow fallback to LLM knowledge | Increases risk of ungrounded answers for out-of-scope queries |

---

## h. Deployment

The application is deployed on **AWS Lambda + Amazon API Gateway (HTTP API v2)** via AWS SAM. The Lambda function runs Python 3.12 with 1024MB memory and a 30-second timeout; a Mangum adapter wraps FastAPI to handle the Lambda event format. API Gateway has CORS enabled and routes all paths to the Lambda. Session data and persistent metrics live in a DynamoDB table (`StudyAssistantSessions`) with PAY_PER_REQUEST billing and 7-day TTL on session items. The embedding index is stored in an S3 bucket (`study-assistant-knowledge-991452971884`) as a gzipped JSON file (~2MB, 38 chunks), downloaded once per cold start. The Lambda execution role has `DynamoDBCrudPolicy`, `S3ReadPolicy`, and `bedrock:InvokeModel` on all Bedrock resources.

The deployment process is three commands:
```bash
python ingest.py          # one-time: embed PDFs, upload to S3
sam build                 # package Lambda with dependencies
sam deploy                # deploy/update CloudFormation stack
```

A few practical constraints are worth noting. Lambda's `/tmp` (512MB) is ephemeral, so JSONL traces reset on cold start, this is why metrics are stored in DynamoDB instead. Cold starts add ~2–3 seconds of latency when the Lambda container is recycled, making the first request after an idle period noticeably slower. The S3 embeddings download (~2MB) only happens on cold start; warm invocations reuse the cached numpy arrays in memory, keeping retrieval fast.

**Public URL:** https://kpu7oyg4pc.execute-api.us-east-1.amazonaws.com

---

## i. Reflection

### What I Learned

Building this project taught me several lessons in building a functional chatbot. Having everything defined in `template.yaml` paid off immediately, I could tear down and redeploy the entire stack in minutes when things broke, and debugging IAM permission errors was far more manageable with SAM's clear error messages than it would have been otherwise. Lambda's size limits were a real constraint I underestimated: ChromaDB's native binaries exceeded the 250MB unzipped limit, forcing a full redesign to pure-Python numpy cosine search. Choosing dependencies for serverless requires checking unzipped deployment size, not just install size.

The evaluation stress tests made clear that chunk granularity directly determines retrieval quality: 400-word chunks worked well for broad conceptual questions but failed on precise numeric lookups. I also learned that system prompt wording has measurable effects on tool selection: small changes like "ALWAYS call lookup_topic first" versus "use tools when relevant" noticeably shifted whether the LLM triggered tool calls for borderline queries. Finally, DynamoDB atomic counters are the right tool for serverless metrics, in-memory counters reset on cold start and file-based logs reset when `/tmp` is recycled, but DynamoDB `ADD` expressions are atomic and persist indefinitely at negligible cost.

### What I Would Improve with More Time

1. **Smaller chunk size (100–150 words)** with sentence-level splitting to preserve specific numeric facts in their own chunks
2. **System prompt refinement**: instruct the LLM to supplement retrieval results with its own knowledge confidently rather than over-hedging
3. **Metadata filtering**: tag chunks by section (Introduction, Results, Algorithm) so queries about specific sections can filter before embedding search
4. **Re-ranking**: add a cross-encoder pass over the top-10 retrieved chunks to improve precision before grounding the LLM
5. **Formal evaluation set**: 20+ prompts with ground-truth expected tool calls and expected answer content; compute retrieval precision@k and tool selection accuracy

### Design Choices I Would Revisit

The 400-word chunk size was the most consequential decision I'd change in hindsight. the stress tests showed it's too coarse, and I would use 100–150 word chunks with sentence-boundary awareness from the start. I would also benchmark the embedding model more carefully; Bedrock Titan v2 is convenient but I'd want to compare it against Cohere or OpenAI embeddings for this specific domain before committing. Finally, I would implement SSE streaming from the beginning rather than retrofitting it, the 4–6 second wait per response is the biggest UX friction point and it compounds for multi-tool turns.

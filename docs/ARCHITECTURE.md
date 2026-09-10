# Architecture and Technical Decisions

This document explains how the system works and, more importantly, *why each choice was made*. It is written so you can answer follow-up questions about any part of the codebase.

---

## 1. Request lifecycle

Trace one request end to end.

**A user uploads `sample_cv.png` and presses "Review my CV".**

1. Streamlit posts the file to `POST /api/upload`.
2. `main.py:upload_cv` validates the extension and size, then calls `tools/ocr.py:extract_text`.
3. OCR dispatches on file type. For a PNG it calls `_ocr_image`, which runs Tesseract. If Tesseract returns fewer than 120 characters, it falls back to Gemini vision.
4. The extracted text goes to `cv_analyst_agent.extract_profile`, one LLM call at temperature 0 that returns structured JSON.
5. The response carries the raw text, the profile, and which OCR engine won. The UI displays all three.
6. The user presses "Review my CV", which posts to `POST /api/analyse` with the profile attached.
7. `graph.run_workflow` builds the initial `CareerState` and invokes the compiled LangGraph with a run config that names the trace and tags it.
8. The **router** node runs one LLM call, returns `{"route": "cv_review", "reasoning": "..."}`, and writes it to state.
9. The conditional edge reads `state["route"]` and jumps to `cv_analyst`.
10. **CV Analyst** builds a retrieval query from the target role, calls `retrieve()`, which embeds the query and runs FAISS search, then makes an LLM call with the retrieved chunks injected as context.
11. A static edge sends execution to `skill_gap`, which calls live search for market requirements *and* retrieval for learning guidance, then makes its own LLM call.
12. Both nodes wrote into `state["findings"]`, merged by the reducer rather than overwriting.
13. The **synthesiser** sees two findings, so it makes a final LLM call merging them into one answer.
14. The API returns the answer plus all evidence. Streamlit renders the answer and three expandable evidence panels.

Total: five LLM calls, one embedding call, one FAISS search, one web search. All of it visible as a nested tree in LangSmith.

---

## 2. Why LangGraph rather than a chain of if-statements

The routing decision could be a Python `if`. Three reasons it is a graph instead:

**Traceability.** A conditional edge appears in the LangSmith trace as a decision node with its inputs and outputs. An `if` statement is invisible to the tracer, so debugging a wrong route means adding print statements.

**State merging.** The `findings` field has a reducer (`merge_findings` in `state.py`). When two nodes both write findings, LangGraph merges them. With manual dispatch you would hand-roll that merge and get it subtly wrong.

**Extensibility.** Adding a sixth specialist means adding one node and one entry to the conditional edge map. With a dispatch function you edit control flow in several places.

---

## 3. Why five agents and not more

The brief warns against inflating agent count. Each agent here owns either a distinct data source or a distinct kind of reasoning:

| Agent | Owns |
|---|---|
| Router | The routing decision. Cheap, temperature 0, no tools. |
| CV Analyst | Document understanding. The only agent that reads raw OCR text. |
| Job Scout | The internet search tool and query construction. |
| Skill Gap | The only agent that combines both data sources for one judgement. |
| Market Intel | The grounding API and its citation metadata. |
| Advisor | Synthesis and playbook-only questions. |

Two agents were considered and rejected. An "ATS scoring agent" would duplicate the CV Analyst's job with a different output format. A "cover letter agent" is a separate feature, not a separate reasoning step, and adding it would have raised the count without deepening the workflow.

---

## 4. The RAG pipeline in detail

### What goes in the knowledge base, and what does not

The rule: **stable and opinionated goes in RAG, volatile goes to search.**

ATS parsing rules do not change monthly. Neither does the structure of a good behavioural interview answer, nor the order in which backend skills should be learned. This is exactly the material a language model gives generic, watered-down answers about, because it averages over every blog post it ever saw. Putting a specific, opinionated version in the knowledge base makes the advice concrete.

Salary figures, open positions, and which companies are hiring change every quarter. Embedding them would guarantee stale answers within months. They go to search and grounding.

### Chunking

`RecursiveCharacterTextSplitter`, `chunk_size=900`, `chunk_overlap=150`.

The splitter tries separators in order: `\n\n`, `\n`, `. `, ` `, `""`. It splits on the largest unit that fits, so a paragraph under 900 characters stays whole and only longer ones get broken at sentence boundaries.

Why 900: large enough that a complete piece of advice (a rule plus its example) survives in one chunk, small enough that four chunks fit in a prompt without crowding out the CV. Chunks of 300 fragment the advice from its example; chunks of 2000 mean retrieving four of them fills the context with mostly irrelevant text.

Why 150 overlap: a rule stated at the end of one chunk and exemplified at the start of the next would otherwise be split. The overlap costs about 17% more storage and eliminates that failure.

Verify the numbers yourself: `python -m backend.rag.ingest --dry-run`.

### Embeddings

`models/text-embedding-004`, 768 dimensions. The same model is used at ingest time and at query time. This is not optional: embedding models define their own vector space, so a query embedded with a different model lands nowhere near its semantically similar chunks and retrieval degrades into noise. `core/llm.py:get_embeddings` is cached so both paths get the identical client.

### Retrieval and injection

`retrieve()` uses `similarity_search_with_score` rather than plain `similarity_search`, because the score is diagnostic: if an answer looks off-topic, the trace shows whether retrieval returned weak matches or whether the LLM ignored good ones.

FAISS returns L2 distance where lower is closer. The UI needs the opposite intuition, so `1 / (1 + distance)` converts it into a rough 0-1 relevance figure.

Chunks are injected as labelled blocks:

```
[Source 1: ATS and CV Formatting Rules]
<chunk text>

---

[Source 2: Skill Roadmaps...]
```

The labels let the model cite which guidance backs a point, and the prompt instructs it to prefer retrieved guidance over its own preferences.

**Query construction matters more than it looks.** The CV Analyst does not retrieve using the user's words. It builds `"CV and resume writing rules, ATS formatting, bullet points for {target_role}"`. Retrieving on "review my CV" would match weakly across every document; the constructed query pulls the ATS and formatting documents specifically.

---

## 5. Search versus grounding

Both are implemented because they are genuinely different mechanisms, and the project uses each where it fits.

### Internet search as a tool (`tools/web_search.py`)

The agent composes a query, the Tavily API returns results, and the agent reasons over the snippets. We control the query, we keep the raw result list, and we can show the user the exact links.

Job Scout uses this because it needs query control. `build_search_query` derives seniority from years of experience and assembles `"{role} {seniority} jobs {location} hiring now"` — a job-board-shaped query that a user would never type.

Provider chain: Tavily if a key exists, otherwise DuckDuckGo, which needs no key. The project therefore runs with one API key total.

### Google Search grounding (`tools/google_grounding.py`)

Gemini is handed the Google Search tool natively. It decides what to look up, searches inside Google's infrastructure, constrains its answer to what it found, and returns `groundingMetadata` containing `web_search_queries` (what it actually searched) and `grounding_chunks` (the sources).

Market Intel uses this because salary figures are the numbers a user acts on. An ungrounded model states a plausible figure from training data with total confidence. Grounding forces the answer to come from retrieved pages and hands back citations we display under **Sources checked**.

If grounding fails, it falls back to the search tool and *labels the answer as ungrounded* rather than degrading silently.

---

## 6. OCR: why three engines

`tools/ocr.py` tries, in order:

1. **`pypdf` text layer.** Most uploaded CVs are digital PDFs with embedded text. Extraction is exact and free. If it yields under 120 characters, the PDF is a scan.
2. **Tesseract.** Local, free, offline. Preprocessing helps: convert to greyscale and apply `ImageOps.autocontrast`, which measurably improves recognition on phone photos where lighting is uneven.
3. **Gemini vision.** Fallback when Tesseract is not installed, or returns near-empty text. Two-column CVs and low-contrast photos routinely defeat Tesseract, and a vision model handles both.

The `MIN_USABLE_CHARS = 120` threshold is what makes the chain work: an engine that "succeeds" while returning 40 characters of garbage has actually failed, and a length check catches that without needing to evaluate quality.

Scanned PDFs are rasterised with `pypdfium2` at `scale=2.0` before OCR. The 2x scale matters — at native resolution small type OCRs badly.

The engine that won is reported in the API response and shown in the UI, so a bad extraction is diagnosable rather than mysterious.

---

## 7. Prompt design

All prompts live in `prompts/templates.py`. Keeping them in one file means they can be reviewed as a set and tuned without hunting through agent code.

Every prompt follows the same shape: **role, task, constraints, output format.**

Specific techniques used:

**Temperature is matched to the task.** Router and extraction run at 0.0, where creativity is a defect. Review and advice run at 0.3-0.4 for readable prose.

**Hallucination is constrained explicitly.** The extraction prompt says never to invent an employer or date and to use `null` instead. The job match prompt says to discuss only roles present in the search results and to say so plainly if there are none. A hallucinated job opening wastes a real person's afternoon.

**Output format is specified exactly.** The router and extractor receive a literal JSON skeleton. `core/parsing.py` then recovers the JSON even when the model wraps it in a markdown fence or adds a preamble — cheaper than retrying the call.

**Negative examples are given where they help.** The CV review prompt asks for a rewritten line the candidate can paste in, not a description of what to change. The synthesis prompt bans preamble and restating the question.

---

## 8. Synthesis, and why it exists

`advisor_agent.synthesise_answer` is the last node on every path.

When only one specialist ran, it passes the finding straight through. This is deliberate: an extra LLM call would only dilute the specialist's specificity, and it saves latency on the common case.

When two ran — the CV review path, which fans into skill gap — it merges. Both specialists will mention the same missing skill, and nobody wants to read it twice. The synthesis prompt requires keeping every concrete number while dropping repetition, and enforces a closing **Do this next** section of at most three actions.

Grounding citations are appended by `_append_source_note` after synthesis, so they survive the merge rather than being summarised away.

---

## 9. Error handling philosophy

Every external call can fail: the search API can rate-limit, grounding can be unavailable, Tesseract may not be installed, the model can return unparseable JSON.

The rule applied throughout: **degrade with an honest label, never fail silently and never crash the request.**

- Router failure defaults to `general_advice` and records the error in state.
- Search returning nothing produces a message saying so, not an invented listing.
- Grounding failure falls back to search and marks the answer as ungrounded.
- Tesseract failure falls through to vision OCR.
- Synthesis failure returns the raw specialist findings, which are still useful.

Errors accumulate in `state["errors"]` and surface in the UI as "Partial failure" warnings, so the user knows which part of the answer to trust less.

---

## 10. LangSmith integration

`core/tracing.py:configure_tracing` translates project settings into the environment variables the LangChain SDK reads. Both the legacy `LANGCHAIN_*` and current `LANGSMITH_*` names are set for version compatibility.

`run_config()` attaches, per request:

- `run_name`: `careerfit:chat` or `careerfit:cv_upload`, which becomes the trace title
- `tags`: `["careerfit-ai", entrypoint]`, for filtering
- `metadata`: the session id, so a conversation can be reconstructed

Because LangGraph and LangChain instrument themselves, every node, LLM call, tool call, and retrieval nests automatically under the top-level run.

**What a trace shows, top to bottom:** the top-level run with the user query → the router node with its prompt and JSON decision → the branch taken → each specialist node, containing its retrieval call with the returned documents and scores, its search or grounding tool call with raw results, and its LLM call with the fully-rendered prompt including injected context → the synthesiser's merge → the final answer.

The rendered prompt is the most useful thing in the trace. It shows exactly which chunks were injected, which is how you diagnose whether a bad answer came from bad retrieval or from the model ignoring good retrieval.

---

## 11. Separation of concerns

The brief asks for separation between API logic, agents, tools, prompts, retrieval, database, and configuration. The mapping:

| Concern | Location | Rule enforced |
|---|---|---|
| API | `backend/main.py` | Validates and shapes only. No prompting or routing logic. |
| Agents | `backend/agents/` | One file per agent. Each is a plain function taking and returning state. |
| Tools | `backend/tools/` | No agent knowledge. Callable and testable standalone. |
| Prompts | `backend/prompts/templates.py` | Strings only. No logic. |
| Retrieval | `backend/rag/` | Owns chunking, embedding, index lifecycle. |
| Config | `backend/config.py` | The only module that reads `os.getenv`. |
| Frontend | `frontend/app.py` | HTTP calls only. No AI logic whatsoever. |

The test of this separation: the agents can be imported and run without a web server (`scripts/smoke_test.py` does exactly that), and the Streamlit UI could be replaced with React without touching a single backend file.

---

## 12. Questions you should be ready for

**Why FAISS over Chroma or Pinecone?** FAISS is a library, not a service: no server to run, no account, one file on disk. For a read-heavy corpus of ~47 chunks that is exactly right. Pinecone would add a network hop and an account dependency for no benefit at this scale. Chroma would be the choice if documents needed adding at runtime.

**Why does the CV review path also run skill gap, but not the reverse?** A review answers "what is wrong"; the user's actual next question is always "so what do I do". Running skill gap after a review pre-empts it. The reverse is not true: someone asking what to learn has not asked for a CV critique.

**What happens with no CV uploaded?** The router checks `has_cv` and reroutes `cv_review` and `skill_gap` requests to `general_advice`, because those branches would otherwise analyse an empty profile.

**Why is extraction done in the API rather than in the graph?** So the user sees their parsed profile immediately and can correct the target role before any analysis runs. It also means the OCR text is interpreted once and reused, rather than re-parsed by each agent.

**How would you handle 1000 concurrent users?** The current design is single-process and stateless per request. The changes needed: move the FAISS index behind a shared service or switch to a hosted vector DB, add a queue for the long-running graph invocations, cache embeddings for repeated queries, and add per-user rate limiting on the grounding calls, which are the most expensive component.

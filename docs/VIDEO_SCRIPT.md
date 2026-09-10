# Video Presentation Script

The brief requires two parts: a demonstration, then a codebase and architecture explanation. Target 18-25 minutes. There is no time limit, but a rambling video is worse than a tight one.

**Record in English. Speak from these notes rather than reading them aloud** — a video that sounds read is penalised on the "can you explain your own work" criterion.

## Before you record

- [ ] `python scripts/smoke_test.py` passes
- [ ] Both processes running, sample CV ready to upload
- [ ] LangSmith open in a browser tab, `careerfit-ai` project, with a **completed run already in it** so you never wait for a trace to appear on camera
- [ ] `.env` file **closed**. Never show it. If your editor is open, close the tab.
- [ ] Do a full dry run once. Note how long each answer takes so you can talk over the waits instead of sitting in silence.

---

# Part 1 — Demonstration (8-10 minutes)

## 1. The problem (90 seconds)

Do not open with the tech stack. Open with the person.

> "A computer science graduate in Dhaka sends out sixty job applications and hears back from two. He does not know why. The advice available to him is generic — write a strong CV, tailor your application — and none of it tells him that his two-column template is being shredded by the employer's applicant tracking system, or that eleven of the eighteen technologies he listed are working against him, or which three skills actually appear in current listings for the roles he wants.
>
> That last part is the hard bit. Even if the advice were specific, the job market changes every quarter. Any advice frozen into a blog post, or into a language model's training data, is already out of date.
>
> CareerFit AI is built for that user: early-career and transitioning tech professionals applying without a recruiter or a mentor."

State why AI is the right tool, since the brief asks for it explicitly:

> "This needs three different kinds of work: reading an unstructured document, comparing it against a target that keeps moving, and forming a judgement. Rules cannot do that, and a static template cannot either. That is also why it is built as separate agents rather than one large prompt."

## 2. Upload and OCR (2 minutes)

Show the UI. Point at the status badges in the sidebar — LLM ready, chunks indexed, search provider, LangSmith project. Say that the frontend checks backend health on load.

Upload `data/sample_cv.png`. **Say clearly that this is an image, not a PDF** — it is the OCR path.

When it completes:

> "Tesseract read about two thousand characters here. If Tesseract had failed — and it does fail on two-column CVs and low-contrast phone photos — the system falls back to a vision model automatically. The UI tells you which engine won, so a bad extraction is diagnosable instead of mysterious."

Expand **Full parsed data**. This is important — show the JSON.

> "The raw OCR text is not what the agents use. A second call parses it into this structured profile, once, and every downstream agent reads this instead of re-interpreting messy text. Notice it picked up eighteen technical skills. Hold that thought."

## 3. CV review — the RAG demonstration (3 minutes)

Set target role `Backend Developer`, location `Dhaka, Bangladesh`. Press **Review my CV**.

While it runs, explain what is happening:

> "The router is classifying this request right now. It will pick the CV review branch, which also chains into skill gap analysis, so two specialists are running and their findings get merged."

When the answer appears, do not read the whole thing. Pick two specific findings and connect them to the source:

> "It flagged the summary line — 'hardworking and passionate individual seeking challenging opportunities' — and rewrote it. It flagged the eighteen-item skills list. And it flagged the date of birth and marital status at the bottom."

**Now expand "What this answer was built from" and open the Knowledge base panel.** This is the single most important moment in the demo:

> "These are the passages retrieved from the vector database before the answer was generated. This one is the ATS and formatting document, relevance 0.7. It contains the rule that a skills section of forty items signals none of them is professional level, and the rule about personality adjectives being unverifiable.
>
> The model did not invent those standards. It retrieved them. That is the difference between this and asking a chatbot to review a CV — the critique comes from a curated knowledge base I control, so I can change the standards by editing a markdown file."

## 4. Live search and grounding (3 minutes)

Press **Find matching jobs**.

> "This is a different branch. The Job Scout agent builds its own search query rather than passing my words through — it derives seniority from years of experience and assembles a job-board-shaped query. Then it scores fit against each real listing."

Expand the **Live search** panel and show the actual URLs.

> "Real links, retrieved just now. The prompt explicitly forbids discussing any role not in these results, because a hallucinated job opening wastes someone's afternoon."

Now press **Salary and demand**, and be explicit about the distinction:

> "This uses a different mechanism, and the difference matters. The last one was search as a tool: my agent composed a query, an API returned snippets, my agent reasoned over them. This one is Google Search grounding — the model itself is bound to Google Search, decides what to look up, and returns citation metadata."

Expand the **Google grounding** panel:

> "These are the searches Google actually ran, and these are the sources backing the answer. Salary figures are the numbers a user will act on in a negotiation, so they need to be traceable rather than recalled from training data."

Scroll to **Sources checked** at the bottom of the answer.

## 5. Wrap Part 1 (30 seconds)

Mention honestly what is not built: no persistence across sessions, job search quality is limited by the search provider, salary coverage for Bangladesh is thinner than for Western markets. The brief invites this, and admitting limits reads as competence.

---

# Part 2 — Code and architecture (10-14 minutes)

Open the repository. Walk the tree in `README.md` first so viewers have a map.

## 6. Architecture overview (2 minutes)

Show the ASCII diagram in the README. Trace one request through it verbally: upload, OCR, extraction, router, branch, specialist, synthesiser, answer.

> "Five specialists. The brief warns against creating agents just to raise a count, so each one owns either a distinct data source or a distinct kind of reasoning. The router owns the decision. The CV Analyst is the only agent that reads raw OCR text. Job Scout owns search. Market Intel owns grounding. Skill Gap is the only one that combines both sources for a single judgement."

Mention what you rejected — it shows deliberate design:

> "I considered an ATS scoring agent and a cover letter agent. The first duplicates the CV Analyst with a different output format. The second is a separate feature, not a separate reasoning step. Neither would have deepened the workflow."

## 7. Folder structure and separation (90 seconds)

Point at the directory tree and name the rule for each:

> "`main.py` validates and shapes, nothing else — no prompting or routing logic in the API layer. Agents are one file each, plain functions taking and returning state. Tools know nothing about agents, so they are testable standalone. Every prompt is in one file. `config.py` is the only module in the project that calls `os.getenv`.
>
> The test of that separation: `scripts/smoke_test.py` runs the whole agent workflow with no web server at all, and the Streamlit UI could be swapped for React without touching a backend file."

## 8. The graph (3 minutes)

Open `backend/agents/graph.py`. The docstring has the diagram.

Walk through `build_graph()`: nodes, entry point, the conditional edge and its route map, the static edge from `cv_analyst` to `skill_gap`, and everything converging on the synthesiser.

Explain why it is a graph:

> "The routing decision could be an if-statement. Three reasons it is not. First, traceability — a conditional edge shows up in LangSmith as a decision node with its inputs and outputs, and an if-statement is invisible to the tracer. Second, state merging — `findings` has a reducer, so when two nodes both write findings LangGraph merges them instead of one clobbering the other. Third, adding a sixth specialist is one node and one map entry."

Open `state.py` and show `merge_findings` and the `Annotated` field.

## 9. RAG pipeline (3-4 minutes)

This is the section evaluators probe hardest. Be precise.

Open `backend/rag/vector_store.py`.

**What is stored:** seven curated career guides, about 35,000 characters. Show the `knowledge_base/` folder.

**The design rule — say this explicitly:**

> "Stable and opinionated goes in RAG. Volatile goes to search. ATS rules do not change monthly, so they are embedded. Salary figures change every quarter, so embedding them would guarantee stale answers within months."

**Chunking:** show `split_documents`. 900 characters, 150 overlap, separators tried in order.

> "Nine hundred is large enough that a rule and its example survive in one chunk, small enough that four chunks fit in a prompt without crowding out the CV. The 150-character overlap means a rule stated at the end of one chunk and exemplified at the start of the next is not split."

Run `python -m backend.rag.ingest --dry-run` on camera. Real numbers are convincing.

**Embeddings:** `text-embedding-004`, 768 dimensions.

> "The same model at ingest time and at query time. That is not optional — embedding models define their own vector space, so a query embedded with a different model lands nowhere near its matching chunks."

**Retrieval:** open `tools/retriever.py`. Show `similarity_search_with_score` and the `1/(1+distance)` conversion. Explain why the score is kept: it tells you whether a bad answer came from weak retrieval or from the model ignoring good retrieval.

**Injection:** show `format_chunks_for_prompt` and the `[Source N: Title]` labels.

**One detail that shows depth** — open `cv_analyst_agent.py` and point at the retrieval query:

> "The CV Analyst does not retrieve using the user's words. It constructs this query from the target role. Retrieving on 'review my CV' matches weakly across every document; this pulls the ATS and formatting documents specifically."

## 10. Tools (2 minutes)

**`tools/ocr.py`** — the three-engine chain and the `MIN_USABLE_CHARS = 120` threshold:

> "An engine that returns forty characters of garbage has failed, even though it did not throw. A length check catches that without needing to evaluate quality."

**`tools/web_search.py` and `tools/google_grounding.py`** — restate the distinction from the demo, now at the code level. Show `_parse_grounded_response` pulling `web_search_queries` and `grounding_chunks` out of the SDK response.

**`prompts/templates.py`** — scroll through. Point out temperature matched to task, the explicit anti-hallucination constraints, and the literal JSON skeletons. Then show `core/parsing.py`:

> "Models sometimes wrap JSON in a markdown fence or add a sentence of preamble. Rather than retrying the call, this recovers the JSON. Cheaper and faster."

## 11. LangSmith (2-3 minutes)

Open `core/tracing.py`. Show `configure_tracing` setting both `LANGCHAIN_*` and `LANGSMITH_*` names, and `run_config` attaching run name, tags, and session metadata.

Switch to your browser and open a completed trace. Expand it fully and narrate top to bottom:

> "Top-level run, the user's query. Router node — here is its prompt, here is the JSON decision and reasoning. It routed to CV review, so execution jumps to the CV Analyst. Inside that node: the retrieval call with the documents it returned and their scores, then the LLM call. If I expand the prompt, I can see the exact chunks that were injected.
>
> That is the most useful thing in the trace. When an answer is wrong, this tells you whether retrieval returned the wrong passages or the model ignored the right ones.
>
> Then the static edge into skill gap, which does both a search call and a retrieval call. Then the synthesiser merging both findings. Every LLM call, tool call and retrieval in the system is here."

Point at token counts and latency per node.

## 12. Error handling and close (90 seconds)

> "Every external call here can fail. The rule throughout is: degrade with an honest label, never fail silently. Router failure defaults to general advice. Search returning nothing produces a message saying so rather than an invented listing. Grounding failure falls back to search and marks the answer as ungrounded. Errors accumulate in state and surface in the UI as partial-failure warnings, so the user knows which part of the answer to trust less."

Close on the problem, not the tech:

> "The point of the architecture is that the advice is specific and traceable. The user can see which passage produced each critique and which source backed each figure. Thank you."

---

## Submission checklist

- [ ] Video uploaded to YouTube, **not** set to private (unlisted or public)
- [ ] GitHub repo pushed, `.env` **not** in it — run `git log --all --full-history -- .env` to be certain
- [ ] `.env.example` present
- [ ] README renders correctly on GitHub
- [ ] 3-4 LangSmith trace links copied, covering: a CV review (RAG), a job match (search), a market intel run (grounding)
- [ ] **Trace links shared** — in LangSmith, open the run, click Share, enable public sharing. An unshared link shows graders a 404.

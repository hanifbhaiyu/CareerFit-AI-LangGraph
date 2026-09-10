# CareerFit AI

A multi-agent career assistant that reads your CV, compares it against live job market data, and tells you what to fix and what to learn next.

Upload a CV as a PDF or a phone photo. The system OCRs it, parses it into a structured profile, routes your question to the right specialist agent, and answers using a combination of vetted career guidance (retrieved from a vector database) and current market information (retrieved from live search and Google Search grounding).

---

## The problem

Job seekers get career advice that is generic, out of date, or both.

A fresh graduate in Dhaka asking "why am I not getting interviews" receives the same advice as a senior engineer in Berlin. Generic advice cannot tell them that their two-column CV template is being mangled by the employer's applicant tracking system, that eleven of the eighteen technologies listed on their CV are hurting rather than helping, or that the three skills actually appearing in current listings for their target role are missing entirely.

Meanwhile, the market data that would answer these questions changes every quarter, and any advice frozen into a blog post or a language model's training data is already stale.

**Target user:** early-career and transitioning technology professionals, initially in Bangladesh, who are applying for jobs without access to a recruiter or mentor.

**Why AI is the right tool here:** the task requires reading an unstructured document, comparing it against a moving external target, and synthesising a judgement. Rules cannot do it, and a static template cannot either. Each of those three steps is a different kind of work, which is why the system is built as separate agents rather than one prompt.

---

## What it does

| Capability | How it works |
|---|---|
| **Reads any CV** | Digital PDFs are parsed directly; scans and photos go through Tesseract OCR, with a vision-model fallback for hard images |
| **Reviews the CV** | Critiques it against ATS and formatting rules retrieved from the knowledge base, not against the model's own opinions |
| **Finds live openings** | Searches the internet for current listings and scores fit for each one |
| **Identifies skill gaps** | Compares the profile against requirements found in real listings, then builds a prioritised 90-day learning plan |
| **Reports salary and demand** | Uses Google Search grounding so figures come with citations rather than from training data |
| **Answers career questions** | Interview prep, negotiation, career transitions, answered from the vetted playbook |

---

## Architecture

```
                            ┌──────────────────┐
   CV upload  ──────────▶   │   OCR pipeline   │  pypdf → Tesseract → vision fallback
                            └────────┬─────────┘
                                     ▼
                            ┌──────────────────┐
                            │  CV extraction   │  raw text → structured profile
                            └────────┬─────────┘
                                     ▼
   User question ──────────▶ ┌──────────────────┐
                             │   ROUTER AGENT   │   one cheap LLM call picks the branch
                             └────────┬─────────┘
          ┌──────────────┬────────────┼─────────────┬────────────────┐
          ▼              ▼            ▼             ▼                ▼
   ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐
   │ CV ANALYST │ │ JOB SCOUT  │ │SKILL GAP │ │MARKET INTEL│ │   ADVISOR    │
   │            │ │            │ │          │ │            │ │              │
   │ RAG        │ │ web search │ │RAG +     │ │ Google     │ │ RAG          │
   │            │ │            │ │search    │ │ grounding  │ │              │
   └──────┬─────┘ └──────┬─────┘ └────┬─────┘ └──────┬─────┘ └───────┬──────┘
          │              │            │              │               │
          └──── ▶ ───────┴────────────┴──────────────┴───────────────┘
                                     ▼
                            ┌──────────────────┐
                            │   SYNTHESISER    │  merges findings into one answer
                            └────────┬─────────┘
                                     ▼
                              Answer + evidence
                     (retrieved chunks, search hits, citations)

   Every node above is traced in LangSmith.
```

A CV review continues automatically into skill gap analysis, because knowing a CV is weak is only half the answer. That is the one path where two specialists run for a single request, which is what the synthesiser exists to merge.

### Agents and their responsibilities

| Agent | Job | Data source |
|---|---|---|
| **Router** | Classify the request into one of five branches | LLM only, temperature 0 |
| **CV Analyst** | Parse OCR text into a profile, then critique it | RAG |
| **Job Scout** | Build a search query, find openings, score fit | Internet search |
| **Skill Gap** | Compare profile to market requirements, build a plan | RAG + internet search |
| **Market Intel** | Salary bands, hiring demand, employer info | Google Search grounding |
| **Advisor** | General career questions, and final synthesis | RAG |

Five specialists, not more. Each one owns a different data source or a different kind of reasoning; none was added to raise a count.

---

## Tech stack

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Orchestration:** LangGraph (`StateGraph` with a conditional router edge)
- **LLM:** Google Gemini via `langchain-google-genai`
- **Embeddings:** `text-embedding-004`, 768 dimensions
- **Vector DB:** FAISS, persisted to disk
- **OCR:** Tesseract (`pytesseract`), with a Gemini vision fallback
- **Search:** Tavily, falling back to DuckDuckGo when no key is set
- **Grounding:** Gemini native Google Search tool via `google-genai`
- **Tracing:** LangSmith

---

## Setup

### 1. Requirements

- Python 3.10 or newer
- Tesseract OCR (optional but recommended)
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: [installer here](https://github.com/UB-Mannheim/tesseract/wiki), then set `TESSERACT_CMD` in `.env`
  - Without it the system falls back to vision OCR automatically

### 2. Install

```bash
git clone <your-repo-url>
cd careerfit-ai

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Then edit `.env`. Only one key is strictly required:

| Variable | Required | Where to get it |
|---|---|---|
| `GOOGLE_API_KEY` | **Yes** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free |
| `LANGSMITH_API_KEY` | For tracing | [smith.langchain.com](https://smith.langchain.com) — free |
| `TAVILY_API_KEY` | Optional | [tavily.com](https://tavily.com) — free tier; falls back to DuckDuckGo if blank |

### 4. Build the vector index

```bash
python -m backend.rag.ingest --force
```

Inspect the chunking without spending API calls:

```bash
python -m backend.rag.ingest --dry-run
```

### 5. Run

```bash
./run.sh
```

Or start the two processes separately:

```bash
uvicorn backend.main:app --reload --port 8000    # terminal 1
streamlit run frontend/app.py                     # terminal 2
```

Open <http://localhost:8501>. The API docs are at <http://127.0.0.1:8000/docs>.

### 6. Verify everything works

```bash
python scripts/smoke_test.py
```

This checks configuration, the vector store, retrieval, search, grounding, OCR, and a full graph run, reporting each independently.

---

## Trying it out

A deliberately flawed sample CV is included at `data/sample_cv.png` (regenerate with `python scripts/make_sample_cv.py`). It contains the exact problems the knowledge base has rules about: a personality-adjective summary, "Responsible for" bullets with no metrics, an eighteen-item skill dump, and personal details that should not be on a CV.

1. Upload it in the sidebar and press **Read this CV**
2. Set target role to `Backend Developer`, location to `Dhaka, Bangladesh`
3. Press **Review my CV**
4. Expand **What this answer was built from** to see which knowledge base passages drove the critique

---

## Project structure

```
careerfit-ai/
├── backend/
│   ├── main.py                  FastAPI app: /upload, /analyse, /health
│   ├── config.py                All settings, read from environment
│   ├── schemas.py               Pydantic request/response models
│   ├── agents/
│   │   ├── graph.py             LangGraph workflow assembly
│   │   ├── state.py             Shared state passed between nodes
│   │   ├── router_agent.py      Supervisor: picks the branch
│   │   ├── cv_analyst_agent.py  OCR text → profile → RAG-backed review
│   │   ├── job_scout_agent.py   Live job search and fit scoring
│   │   ├── skill_gap_agent.py   RAG + search → 90-day plan
│   │   ├── market_intel_agent.py Google Search grounding
│   │   └── advisor_agent.py     General advice + final synthesis
│   ├── tools/
│   │   ├── ocr.py               pypdf → Tesseract → vision fallback
│   │   ├── web_search.py        Tavily → DuckDuckGo
│   │   ├── google_grounding.py  Gemini native Search grounding
│   │   └── retriever.py         Semantic search over the knowledge base
│   ├── rag/
│   │   ├── vector_store.py      Chunking, embedding, FAISS index
│   │   ├── ingest.py            CLI to build the index
│   │   └── knowledge_base/      7 curated career guides
│   ├── prompts/templates.py     Every prompt, in one file
│   └── core/
│       ├── llm.py               Model factories
│       ├── tracing.py           LangSmith wiring
│       ├── parsing.py           Robust JSON extraction from LLM output
│       └── logging_config.py
├── frontend/app.py              Streamlit UI
├── scripts/
│   ├── smoke_test.py            End-to-end verification
│   └── make_sample_cv.py        Generates the demo CV
├── docs/
│   ├── ARCHITECTURE.md          Detailed technical walkthrough
│   └── VIDEO_SCRIPT.md          Presentation outline
├── .env.example
└── requirements.txt
```

---

## How retrieval works

**What is stored:** seven curated career guides covering ATS rules, CV writing, interview preparation, salary negotiation, skill roadmaps, career transitions, and the Bangladesh technology job market. About 35,000 characters total.

**Chunking:** `RecursiveCharacterTextSplitter`, 900-character chunks with 150 characters of overlap, producing roughly 47 chunks. The splitter tries paragraph breaks first, then line breaks, then sentences, so chunks rarely cut mid-sentence. The overlap means a fact sitting on a boundary still appears intact in one chunk.

**Embeddings:** Google `text-embedding-004` produces a 768-dimension vector per chunk. The same model embeds the query, which is essential — using different models at ingest and query time puts the vectors in different spaces and retrieval returns noise.

**Retrieval:** the query is embedded and FAISS runs a nearest-neighbour search, returning the top 4 chunks with distance scores. The API converts distance into a 0-1 relevance figure for display.

**Injection:** retrieved chunks are formatted as labelled `[Source N: Title]` blocks and inserted into the prompt, with an instruction to prefer the retrieved guidance over the model's own preferences.

**The design decision worth noting:** stable, opinionated advice comes from RAG; volatile facts like salaries and open roles come from search and grounding. Mixing them would mean either stale salary figures or ungrounded advice.

---

## Search versus grounding

The project implements both, because they are different mechanisms:

**Internet search** (`tools/web_search.py`) is a tool call. The agent composes a query, an external API returns results, and the agent reasons over the snippets. We control the query and keep the raw results, which is what the Job Scout needs.

**Google Search grounding** (`tools/google_grounding.py`) binds the model itself to Google Search. Gemini decides what to look up, runs it inside Google's infrastructure, constrains its answer to what it found, and returns `groundingMetadata` listing the sources and the queries it ran. Market Intel uses this because salary figures are the numbers users act on, so they need citations.

---

## LangSmith tracing

Set `LANGSMITH_TRACING=true` and add `LANGSMITH_API_KEY` in `.env`. Every run then appears at [smith.langchain.com](https://smith.langchain.com) under the `careerfit-ai` project.

Each trace shows the full execution tree: the router's decision and reasoning, the branch taken, every LLM call with its prompt and response, every tool call including retrieval and search, and the synthesiser's merge. Runs are tagged with the entrypoint (`chat` or `cv_upload`) and carry the session id in metadata, so a whole conversation can be reconstructed.

---

## Environment variables

See `.env.example` for the full annotated list. `.env` is gitignored and no key is ever hard-coded — everything is read through `backend/config.py`.

---

## Known limits

- Job search quality depends on the search provider. DuckDuckGo returns fewer job-board results than Tavily; a dedicated jobs API would be the right upgrade.
- Tesseract struggles with two-column CVs and low-contrast phone photos. The vision fallback handles most of these, at higher cost per request.
- There is no persistence across sessions. Profiles live in Streamlit session state and are lost on refresh.
- Salary figures are only as current as what search returns, and coverage for the Bangladesh market is thinner than for the US or Europe.



**GitHub: https://github.com/hanifbhaiyu/careerfit-ai_C223403

YouTube: 

Trace 1 - CV Review (RAG + multi-agent):

https://smith.langchain.com/public/d505e719-b1de-45cf-ac2a-6e072d4cec0b/r/01a077c2-b94c-71b1-8784-02a9edc6ae8a?start_time=2026-09-06T17%3A27%3A25.260131Z

Trace 2 - Job Match (internet search):

https://smith.langchain.com/public/d754c031-8a96-4488-b23b-674082b74cde/r/01a07804-bcb4-7293-bb96-dd4d56cd3f71?start_time=2026-09-06T18%3A39%3A31.508963Z

Trace 3 - Salary (Google grounding):

https://smith.langchain.com/public/34247fe4-ceb8-43c4-89ce-3be2a08883b1/r/01a07805-ffeb-7221-8c24-d9f056a6783c?start_time=2026-09-06T18%3A40%3A54.251627Z
**

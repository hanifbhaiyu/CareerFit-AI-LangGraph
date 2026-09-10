"""CareerFit AI - Streamlit front end.

Talks to the FastAPI backend over HTTP only. It holds no prompting, retrieval or
agent logic, which is the point: the same backend could serve a React client
without any change.

User journey:
    1. Upload a CV, which is OCR'd and parsed into a profile the user can see.
    2. State a target role and location.
    3. Ask a question, or use a starter action.
    4. Read the answer, and expand the evidence panels to see the retrieved
       passages, search results and grounding citations behind it.
"""

from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 360

st.set_page_config(
    page_title="CareerFit AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling deliberately avoids setting background or text colours. Streamlit
# ships light and dark themes, and forcing one colour while the other theme
# supplies the opposite text colour makes content invisible. Only the accent
# tag sets both colours together, so it stays readable either way.
st.markdown(
    """
    <style>
      .cf-tag {
        display: inline-block; padding: 2px 10px; margin: 0 6px 4px 0;
        border: 1px solid #7FA8A0; border-radius: 3px;
        background: #24504C; color: #EEF4F2; font-size: 0.78rem;
      }
      .cf-meta { opacity: 0.75; font-size: 0.85rem; }
      .stButton button { border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

def init_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "cv_text": "",
        "cv_profile": {},
        "ocr_engine": "",
        "ocr_chars": 0,
        "history": [],
        "health": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


# --------------------------------------------------------------------------- #
# API calls
# --------------------------------------------------------------------------- #

def fetch_health() -> dict | None:
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def upload_cv(file) -> dict | None:
    try:
        response = requests.post(
            f"{API_BASE}/api/upload",
            files={"file": (file.name, file.getvalue())},
            data={"session_id": st.session_state.session_id},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text)
            st.error(f"Upload failed: {detail}")
            return None
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {API_BASE}. {exc}")
        return None


def analyse(query: str, target_role: str, location: str) -> dict | None:
    payload = {
        "query": query,
        "session_id": st.session_state.session_id,
        "cv_text": st.session_state.cv_text,
        "cv_profile": st.session_state.cv_profile or None,
        "target_role": target_role,
        "location": location,
    }
    try:
        response = requests.post(
            f"{API_BASE}/api/analyse", json=payload, timeout=REQUEST_TIMEOUT
        )
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text)
            st.error(f"Analysis failed: {detail}")
            return None
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {API_BASE}. {exc}")
        return None


# --------------------------------------------------------------------------- #
# Sidebar: configuration, upload, profile
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### CareerFit AI")
    st.markdown(
        '<p class="cf-meta">Upload a CV, name a target role, get a grounded plan.</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.health is None:
        st.session_state.health = fetch_health()

    health = st.session_state.health
    if health is None:
        st.error(
            f"Backend not reachable at {API_BASE}.\n\n"
            "Start it with:\n`uvicorn backend.main:app --reload`"
        )
    else:
        badges = []
        badges.append("LLM ready" if health.get("llm_configured") else "LLM key missing")
        if health.get("vector_store_ready"):
            badges.append(f"{health.get('indexed_chunks', 0)} chunks indexed")
        else:
            badges.append("index not built")
        badges.append(f"search: {health.get('search_provider')}")
        badges.append(
            f"LangSmith: {health.get('langsmith_project')}"
            if health.get("langsmith_tracing")
            else "LangSmith off"
        )
        st.markdown(
            " ".join(f'<span class="cf-tag">{badge}</span>' for badge in badges),
            unsafe_allow_html=True,
        )

        for problem in health.get("problems", []):
            st.warning(problem)

    st.divider()

    st.markdown("#### Your CV")
    uploaded = st.file_uploader(
        "PDF, image, or plain text",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md"],
        help="Scanned images and photos are read with OCR.",
    )

    if uploaded is not None and st.button("Read this CV", use_container_width=True):
        with st.spinner("Running OCR and parsing the profile..."):
            result = upload_cv(uploaded)
        if result:
            st.session_state.cv_text = result["cv_text"]
            st.session_state.cv_profile = result["cv_profile"]
            st.session_state.ocr_engine = result["ocr_engine"]
            st.session_state.ocr_chars = result["characters_extracted"]
            st.success(
                f"Read {result['characters_extracted']:,} characters "
                f"using {result['ocr_engine']}."
            )

    if st.session_state.cv_profile:
        profile = st.session_state.cv_profile
        st.markdown("#### Parsed profile")
        if profile.get("name"):
            st.markdown(f"**{profile['name']}**")
        if profile.get("headline"):
            st.caption(profile["headline"])
        if profile.get("years_experience") is not None:
            st.caption(f"{profile['years_experience']} years of experience")

        skills = profile.get("technical_skills") or []
        if skills:
            st.markdown(
                " ".join(f'<span class="cf-tag">{skill}</span>' for skill in skills[:14]),
                unsafe_allow_html=True,
            )

        with st.expander("Full parsed data"):
            st.json(profile)
        with st.expander(f"Raw OCR text ({st.session_state.ocr_engine})"):
            st.text(st.session_state.cv_text[:4000])

        if st.button("Clear CV", use_container_width=True):
            st.session_state.cv_text = ""
            st.session_state.cv_profile = {}
            st.rerun()

    st.divider()
    st.caption(f"Session `{st.session_state.session_id[:8]}`")


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #

st.title("Where you stand, and what to do next")
st.markdown(
    '<p class="cf-meta">Five specialists read your CV against live job market data: '
    "a router picks who works on your question, and their findings are merged into one answer.</p>",
    unsafe_allow_html=True,
)

col_role, col_location = st.columns(2)
with col_role:
    target_role = st.text_input(
        "Target role",
        value="Backend Developer",
        help="The role you are aiming for. Drives job search and gap analysis.",
    )
with col_location:
    location = st.text_input("Location", value="Dhaka, Bangladesh")

st.markdown("##### Start with")
starters = {
    "Review my CV": "Review my CV and tell me what is holding it back.",
    "Find matching jobs": "Find live job openings that match my profile and score my fit.",
    "What should I learn": "What skills am I missing for my target role, and in what order should I learn them?",
    "Salary and demand": "What is the current salary range and hiring demand for my target role?",
}

starter_cols = st.columns(len(starters))
chosen_starter = None
for column, (label, prompt) in zip(starter_cols, starters.items()):
    if column.button(label, use_container_width=True):
        chosen_starter = prompt

typed_query = st.text_area(
    "Or ask your own question",
    placeholder="How do I explain a two-year gap in my CV to an interviewer?",
    height=90,
)

submitted = st.button("Get advice", type="primary", use_container_width=True)
query = chosen_starter or (typed_query if submitted else None)

if query:
    if not st.session_state.cv_profile and "my CV" in query:
        st.info("No CV uploaded, so the answer will be general. Upload one in the sidebar for specifics.")

    with st.spinner("Routing to specialists, retrieving guidance, searching the market..."):
        result = analyse(query, target_role, location)

    if result:
        st.session_state.history.insert(0, {"query": query, "result": result})

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

for index, entry in enumerate(st.session_state.history):
    result = entry["result"]
    st.divider()
    st.markdown(f"**You asked:** {entry['query']}")

    route_label = result.get("route", "unknown").replace("_", " ")
    st.markdown(
        f'<span class="cf-tag">routed to: {route_label}</span>'
        f'<span class="cf-meta">{result.get("route_reasoning", "")}</span>',
        unsafe_allow_html=True,
    )

    # Rendered inside a container rather than a raw HTML div, because markdown
    # in the answer (headings, bold, links) would not be parsed inside one.
    with st.container(border=True):
        st.markdown(result.get("answer", ""))

    for error in result.get("errors", []):
        st.warning(f"Partial failure: {error}")

    chunks = result.get("retrieved_chunks") or []
    searches = result.get("search_results") or []
    sources = result.get("grounding_sources") or []

    if chunks or searches or sources:
        st.markdown("##### What this answer was built from")
        evidence_cols = st.columns(3)

        with evidence_cols[0]:
            with st.expander(f"Knowledge base ({len(chunks)})"):
                for chunk in chunks:
                    st.markdown(f"**{chunk['title']}**  \n`{chunk['source']}` · relevance {chunk.get('score')}")
                    st.caption(chunk["preview"] + "...")

        with evidence_cols[1]:
            with st.expander(f"Live search ({len(searches)})"):
                for hit in searches:
                    st.markdown(f"[{hit['title']}]({hit['url']})")
                    st.caption(hit["snippet"][:200] + "...")

        with evidence_cols[2]:
            with st.expander(f"Google grounding ({len(sources)})"):
                queries = result.get("grounding_queries") or []
                if queries:
                    st.caption("Searches Google ran: " + "; ".join(queries))
                for source in sources:
                    st.markdown(f"[{source['title']}]({source['url']})")

    if index == 0 and len(st.session_state.history) > 1:
        st.caption("Earlier answers continue below.")

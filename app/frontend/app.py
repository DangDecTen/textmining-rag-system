"""
Streamlit chat UI for the RAG system.

Talks to the FastAPI backend (app/backend/api.py) over HTTP -- it never
imports src/ directly, so it can be deployed/scaled independently of the
backend.

Run (in a separate terminal from the API):
    python -m streamlit run app/frontend/app.py
"""
import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000",)

st.set_page_config(page_title="MITRE ATT&CK Assistant", page_icon="🛡️", layout="wide")

@st.cache_data(ttl=60)
def get_available_options() -> tuple[list[str], list[str], list[str]]:
    """Asks the API what's registered (src/retrieval/registry.py,
    src/generation/registry.py) instead of hardcoding the list here. Add a
    new retriever/generator with @register_retriever / @register_generator
    and it shows up in this dropdown automatically -- no frontend change."""
    try:
        resp = requests.get(f"{API_URL}/", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("available_retrievers", ["bm25"]), data.get("available_generators", ["llama"]), data.get("available_prompts", ["baseline"])
    except requests.RequestException:
        return ["bm25", "dense"], ["llama", "qwen"], ["baseline"]


retrievers, generators, prompts = get_available_options()

with st.sidebar:
    st.title("⚙️ Configuration")
    retriever_type = st.selectbox("Retriever", retrievers, help="Retrieval method used to find relevant ATT&CK documents.")
    generator_type = st.selectbox("Generator", generators, help="LLM used to generate the final answer.")
    prompt_type = st.selectbox("Prompt", prompts, help="Prompt strategy used to instruct the generator.")
    st.divider()
    st.caption(f"API: {API_URL}")

st.title("🛡️ MITRE ATT&CK Assistant")
st.markdown("Ask a question about the MITRE ATT&CK knowledge base and get a grounded answer with supporting sources.")
# --------- Chat history ---------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander("📚 Sources"):
        for c in citations:
            label = c.get("subject_name") or c.get("subject_id", "Unknown subject")
            st.markdown(f"**{label}** — {c.get('source', '')}")
            if c.get("relation_name"):
                st.caption(f"Related: {c['relation_name']}")
            if c.get("url"):
                st.markdown(c["url"])
            st.divider()


def render_retrieved_context(retrieved_context: list[dict]) -> None:
    if not retrieved_context:
        return
    with st.expander(f"🔍 Retrieved chunks ({len(retrieved_context)})", expanded=False):
        for r in retrieved_context:
            doc = r.get("document") or {}
            with st.expander(f"{r.get('doc_id', 'unknown')} | Score: {r.get('score', 0):.3f}"):
                st.write(doc.get("text", "(document text unavailable)"))

def render_metrics(data: dict) -> None:
    """Render compact metadata about the query."""
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Retriever",
        data.get("retriever", "—"),
    )

    col2.metric(
        "Generator",
        data.get("generator", "—"),
    )

    col3.metric(
        "Retrieved",
        len(data.get("retrieved_context", [])),
    )

    latency = data.get("latency_ms")

    if latency is not None:
        col4.metric(
            "Latency",
            f"{latency / 1000:.2f}s",
        )
    else:
        col4.metric("Latency", "—")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            metadata = msg.get("metadata")
            if metadata:
                with st.caption(
                    f"{metadata.get('retriever', '—')} · "
                    f"{metadata.get('generator', '—')} · "
                    f"{metadata.get('prompt', '—')}"
                ):
                    pass

            render_citations(msg.get("citations", []))
            render_retrieved_context(msg.get("retrieved_context", []))

# --------- User input ---------
question = st.chat_input("Ask a cybersecurity question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Processing query...", expanded=True)
        status.write(f"🔍 Retrieving  with **{retriever_type}**...")
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "query": question,
                    #"k": top_k,
                    "retriever": retriever_type,
                    "generator": generator_type,
                    "prompt": prompt_type,
                },
                timeout=120,
            )
            response.raise_for_status()
            status.write(f"✍️ Generating answer with **{generator_type}** using **{prompt_type}** prompt...")
            data = response.json()
            answer = data["answer"]
            abstained = data["abstained"]
            citations = data["citations"]
            retrieved_context = data["retrieved_context"]
            status.update(label="✅ Done", state="complete")

        except requests.RequestException as e:
            status.update(label="❌ Request failed", state="error")
            st.error(f"Could not reach the API at {API_URL}. Is it running? ({e})")
            st.stop()
        except (KeyError, ValueError) as e:
            status.update(label="❌ Unexpected response", state="error")
            st.error(f"Unexpected API response shape: {e}")
            st.stop()

        # --------- Answer ---------
        if abstained:
            st.warning(
                "The system could not find enough information "
                "in the retrieved context to answer this question."
            )
        st.markdown(answer)

    #     if not abstained and retrieved_context:
    #         col1, col2, col3 = st.columns(3)
    #         col1.metric("Retriever", retriever_type)
    #         col2.metric("Chunks Retrieved", len(retrieved_context))
    #         col3.metric("Top Score", f"{retrieved_context[0]['score']:.3f}")

    #     render_citations(citations)
    #     render_retrieved_context(retrieved_context)

        st.divider()

        render_metrics(data)

        # Show the exact configuration used.
        with st.expander("⚙️ Query configuration"):
            st.write(
                {
                    "Retriever": data.get(
                        "retriever",
                        retriever_type,
                    ),
                    "Generator": data.get(
                        "generator",
                        generator_type,
                    ),
                    "Prompt": data.get(
                        "prompt",
                        prompt_type,
                    ),
                }
            )

        # ---------------------------------------------------------------
        # Sources / debug
        # ---------------------------------------------------------------

        render_citations(citations)

        render_retrieved_context(
            retrieved_context
        )

    # ---------------------------------------------------------------
    # Save assistant message
    # ---------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "retrieved_context": retrieved_context,
            "metadata": {
                "retriever": data.get(
                    "retriever",
                    retriever_type,
                ),
                "generator": data.get(
                    "generator",
                    generator_type,
                ),
                "prompt": data.get(
                    "prompt",
                    prompt_type,
                ),
            },
        }
    )

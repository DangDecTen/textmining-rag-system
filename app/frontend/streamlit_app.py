"""
Streamlit chat UI for the RAG system.

Talks to the FastAPI backend (app/backend/api.py) over HTTP -- it never
imports src/ directly, so it can be deployed/scaled independently of the
backend.

Run (in a separate terminal from the API):
    python -m streamlit run app/frontend/streamlit_app.py
"""
import os
from pathlib import Path
import requests
import streamlit as st
from app.backend.app_prompts import prompt_label


API_URL = os.getenv("API_URL", "http://localhost:8000",)

st.set_page_config(page_title="MITRE ATT&CK Assistant", page_icon="🛡️", layout="wide")

def load_css():
    css_path = Path(__file__).parent / "styles.css"

    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )

load_css()

RESPONSE_MODES: dict[str, dict[str, str]] = {
    "quick": {
        "label": "⚡ Quick",
        "description": "Fast, concise answers for straightforward questions.",
        "retriever": "bm25",
        "generator": "gpt",
        "prompt": "baseline",
    },
    "balanced": {
        "label": "🎯 Balanced",
        "description": "Clear answers with supporting evidence.",
        "retriever": "hybrid",
        "generator": "qwen",
        "prompt": "evidence",
    },
    "deep": {
        "label": "🔎 Deep Analysis",
        "description": "More detailed analysis with evidence checking.",
        "retriever": "colbert",
        "generator": "qwen",
        "prompt": "cot_verification",
    },
}
CUSTOM_MODE = "custom"

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
        return data.get("available_retrievers", ["bm25"]), data.get("available_generators", ["qwen"]), data.get("available_prompts", ["baseline"])
    except requests.RequestException:
        return ["bm25", "dense"], ["gpt", "qwen"], ["baseline"]


retrievers, generators, prompts = get_available_options()

# =============================================================================
# Session state
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "response_mode" not in st.session_state:
    st.session_state.response_mode = "balanced"

if "custom_retriever" not in st.session_state:
    st.session_state.custom_retriever = "hybrid"

if "custom_generator" not in st.session_state:
    st.session_state.custom_generator = "qwen"

if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = "evidence"

# =============================================================================
# Helpers: response mode
# =============================================================================

def get_config_for_mode(mode):
    """Return the actual API configuration for the selected mode."""

    if mode in RESPONSE_MODES:
        preset = RESPONSE_MODES[mode]

        return {
            "retriever": preset["retriever"],
            "generator": preset["generator"],
            "prompt": preset["prompt"],
        }

    # Only Custom mode uses the manually selected configuration.
    return {
        "retriever": st.session_state.custom_retriever,
        "generator": st.session_state.custom_generator,
        "prompt": st.session_state.custom_prompt,
    }


def get_mode_description(mode):
    if mode == CUSTOM_MODE:
        return "Your manually selected RAG configuration."

    return RESPONSE_MODES[mode]["description"]


def get_mode_label(mode):
    if mode == CUSTOM_MODE:
        return "⚙ Custom"

    return RESPONSE_MODES[mode]["label"]


def get_config_description(config):
    return (
        f"{config['retriever']} retrieval · "
        f"{config['generator']} · "
        f"{prompt_label(config['prompt'])}"
    )


def detect_preset(config):
    """
    If the current custom configuration exactly matches a preset,
    return that preset. Otherwise return Custom.
    """

    for mode, preset in RESPONSE_MODES.items():
        if (
            config["retriever"] == preset["retriever"]
            and config["generator"] == preset["generator"]
            and config["prompt"] == preset["prompt"]
        ):
            return mode

    return CUSTOM_MODE

# =============================================================================
# Helpers: source/evidence
# =============================================================================

def find_citation(doc_id, citations):
    for citation in citations:
        if citation.get("doc_id") == doc_id:
            return citation

    return None


def render_source(citation):
    if not citation:
        return

    url = citation.get("url")

    if not url:
        return

    source = citation.get("source") or "source"

    st.markdown(
        f'<a class="source-link" href="{url}" target="_blank">'
        f"↗ Open {source} source"
        f"</a>",
        unsafe_allow_html=True,
    )


def render_evidence_card(retrieval, citation, index):
    document = retrieval.get("document") or {}

    doc_id = retrieval.get(
        "doc_id",
        document.get("doc_id", "unknown"),
    )

    score = retrieval.get("score")
    

    subject_name = (
        document.get("subject_name")
        or (citation or {}).get("subject_name")
        or document.get("subject_id")
        or (citation or {}).get("subject_id")
        or "ATT&CK knowledge"
    )

    subject_id = (
        document.get("subject_id")
        or (citation or {}).get("subject_id")
    )

    text = document.get(
        "text",
        "(Retrieved text unavailable.)",
    )

    source = (
        document.get("source")
        or (citation or {}).get("source")
        or "Unknown source"
    )

    subject_type = document.get("subject_type")

    field = (
        document.get("field")
        or (citation or {}).get("field")
    )

    relation_name = (
        document.get("relation_name")
        or (citation or {}).get("relation_name")
    )

    with st.container(border=True):

        header_col, score_col = st.columns([5, 1])

        with header_col:

            st.markdown(f"**Evidence {index}**")

            title = subject_name

            if subject_id:
                title += f" · `{subject_id}`"

            st.markdown(f"### {title}")

        with score_col:

            if score is not None:
                st.markdown(
                    f"""
                    <div class="relevance-score">
                        <div class="relevance-score-label">Relevance</div>
                        <div class="relevance-score-value">{score:.3f}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                )

        with st.expander("📄 Retrieved passage", expanded=False):
            quoted_text = text.replace("\n", "\n> ")
            st.markdown(
                f"> {quoted_text}"
            )

        metadata = [source]

        if subject_type:
            metadata.append(subject_type)

        if field:
            metadata.append(field)

        if relation_name:
            metadata.append(
                f"Related: {relation_name}"
            )

        st.caption(" · ".join(metadata))

        render_source(citation)

def _render_evidence_contents(
    retrieved_context,
    citations,
):
    visible = retrieved_context[:3]

    for index, retrieval in enumerate(
        visible,
        start=1,
    ):
        doc_id = retrieval.get("doc_id")

        citation = find_citation(
            doc_id,
            citations,
        )

        render_evidence_card(
            retrieval,
            citation,
            index,
        )

    remaining = retrieved_context[3:]

    if remaining:

        with st.expander(
            f"🔍 Show {len(remaining)} additional retrieved "
            f"{'passage' if len(remaining) == 1 else 'passages'}"
        ):

            for index, retrieval in enumerate(
                remaining,
                start=4,
            ):

                doc_id = retrieval.get("doc_id")

                citation = find_citation(
                    doc_id,
                    citations,
                )

                render_evidence_card(
                    retrieval,
                    citation,
                    index,
                )

def render_evidence(retrieved_context, citations, expanded=False):
    if not retrieved_context:
        return

    count = len(retrieved_context)

    label = (
        f"📚 Evidence used for this answer "
        f"({count} {'passage' if count == 1 else 'passages'})"
    )

    if expanded:

        st.markdown(
            '<div class="evidence-heading">📚 Evidence used for this answer</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="evidence-subtitle">'
            f"{count} "
            f"{'retrieved passage' if count == 1 else 'retrieved passages'}"
            f'</div>',
            unsafe_allow_html=True,
        )

        _render_evidence_contents(
            retrieved_context,
            citations,
        )

    else:

        with st.expander(
            label,
            expanded=False,
        ):
            _render_evidence_contents(
                retrieved_context,
                citations,
            )


# =============================================================================
# Helpers: technical details
# =============================================================================
def render_technical_details(data, mode, config):

    with st.expander("⚙️ Technical details"):

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Response mode**")
            st.write(get_mode_label(mode))

            st.write("**Retriever**")
            st.code(
                data.get(
                    "retriever",
                    config["retriever"],
                )
            )

            st.write("**Prompt strategy**")
            st.write(
                prompt_label(
                    data.get(
                        "prompt",
                        config["prompt"],
                    )
                )
            )

        with col2:

            st.write("**Generator**")
            st.code(
                data.get(
                    "generator",
                    config["generator"],
                )
            )

            retrieved = data.get(
                "retrieved_context",
                [],
            )

            st.write("**Retrieved passages**")
            st.write(len(retrieved))

            latency = data.get("latency_ms")

            if latency is not None:
                st.write("**Latency**")
                st.write(
                    f"{latency / 1000:.2f} s"
                )

        prompt_tokens = data.get(
            "prompt_tokens"
        )

        completion_tokens = data.get(
            "completion_tokens"
        )

        if (
            prompt_tokens is not None
            or completion_tokens is not None
        ):

            with st.expander(
                "Generation statistics"
            ):

                if prompt_tokens is not None:
                    st.write(
                        f"Prompt tokens: "
                        f"**{prompt_tokens:,}**"
                    )

                if completion_tokens is not None:
                    st.write(
                        f"Completion tokens: "
                        f"**{completion_tokens:,}**"
                    )


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:

    st.markdown("## 🛡️ ATT&CK Assistant")

    st.caption("Grounded cybersecurity answers with inspectable evidence.")

    st.divider()

    st.markdown("### Response mode")

    mode_options = [
        "quick",
        "balanced",
        "deep",
        CUSTOM_MODE,
    ]

    mode_labels = {
        "quick": "⚡ Quick",
        "balanced": "🎯 Balanced",
        "deep": "🔎 Deep Analysis",
        CUSTOM_MODE: "⚙ Custom",
    }

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # The selectbox itself owns the current widget state.
    # We do not call st.rerun() here.
    # -------------------------------------------------------------------------

    selected_mode = st.selectbox(
        "Response mode",
        options=mode_options,
        format_func=lambda x: mode_labels[x],
        index=mode_options.index(
            st.session_state.response_mode
        ),
        key="response_mode_selector",
        label_visibility="collapsed",
    )

    st.session_state.response_mode = selected_mode

    current_config = get_config_for_mode(
        selected_mode
    )

    st.markdown(
        f"""
        <div class="mode-description">
            <div class="mode-description-title">
                {get_mode_label(selected_mode)}
            </div>
            <div class="mode-description-text">
                {get_mode_description(selected_mode)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        get_config_description(current_config)
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Custom configuration
    # -------------------------------------------------------------------------

    if selected_mode == CUSTOM_MODE:

        with st.expander(
            "⚙ Configure RAG",
            expanded=True,
        ):

            st.caption(
                "Select the individual retrieval, generation, "
                "and prompting components."
            )

            custom_retriever = st.selectbox(
                "Retriever",
                retrievers,
                index=(
                    retrievers.index(
                        st.session_state.custom_retriever
                    )
                    if st.session_state.custom_retriever in retrievers
                    else 0
                ),
                key="custom_retriever_selector",
                help=(
                    "Method used to retrieve relevant "
                    "ATT&CK knowledge."
                ),
            )

            custom_generator = st.selectbox(
                "Generator",
                generators,
                index=(
                    generators.index(
                        st.session_state.custom_generator
                    )
                    if st.session_state.custom_generator in generators
                    else 0
                ),
                key="custom_generator_selector",
                help=(
                    "Language model used to generate "
                    "the final answer."
                ),
            )

            custom_prompt = st.selectbox(
                "Prompt strategy",
                prompts,
                index=(
                    prompts.index(
                        st.session_state.custom_prompt
                    )
                    if st.session_state.custom_prompt in prompts
                    else 0
                ),
                format_func=prompt_label,
                key="custom_prompt_selector",
                help=(
                    "Prompt strategy used to instruct "
                    "the generator."
                ),
            )

            st.session_state.custom_retriever = custom_retriever
            st.session_state.custom_generator = custom_generator
            st.session_state.custom_prompt = custom_prompt

            st.caption(
                get_config_description(
                    {
                        "retriever": custom_retriever,
                        "generator": custom_generator,
                        "prompt": custom_prompt,
                    }
                )
            )

    st.divider()

    st.markdown("### System")
    st.markdown(
        '<div class="api-status">'
        '<span class="api-status-dot"></span>'
        '<span>API configured</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(API_URL)


# =============================================================================
# Main page
# =============================================================================

st.markdown("# 🛡️ MITRE ATT&CK Assistant")
st.markdown(
    "Ask a cybersecurity question and explore the evidence behind the answer."
)


# =============================================================================
# Empty state
# =============================================================================

if not st.session_state.messages:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            **🎯 Grounded answers**

            Answers are generated from retrieved
            AttackQA knowledge.
            """
        )

    with col2:
        st.markdown(
            """
            **📚 Inspect the evidence**

            See the passages retrieved for each
            answer and open their sources.
            """
        )

    with col3:
        st.markdown(
            """
            **🔬 Experiment**

            Use Custom mode to test different
            RAG configurations.
            """
        )


# =============================================================================
# Current mode
# =============================================================================

active_mode = st.session_state.response_mode

active_config = get_config_for_mode(
    active_mode
)

st.markdown("### Response mode")

mode_col1, mode_col2 = st.columns(
    [1, 4]
)

with mode_col1:

    st.markdown(
        f"""
        <div class="current-mode-badge">
            {get_mode_label(active_mode)}
        </div>
        """,
        unsafe_allow_html=True,
    )

with mode_col2:

    st.markdown(
        f"**{get_mode_description(active_mode)}**"
    )

    st.caption(
        get_config_description(active_config)
    )


# =============================================================================
# Render session transcript
# =============================================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )
        if message["role"] != "assistant":
            continue

        metadata = (
            message.get("metadata")
            or {}
        )

        saved_mode = metadata.get(
            "mode",
            "balanced",
        )

        saved_config = {
            "retriever": metadata.get(
                "retriever",
                "—",
            ),
            "generator": metadata.get(
                "generator",
                "—",
            ),
            "prompt": metadata.get(
                "prompt",
                "baseline",
            ),
        }

        retrieved_context = (
            message.get(
                "retrieved_context"
            )
            or []
        )

        citations = (
            message.get(
                "citations"
            )
            or []
        )

        if message.get(
            "abstained",
            False,
        ):

            st.warning(
                "There was not enough supporting evidence "
                "in the retrieved knowledge to answer this "
                "question reliably."
            )

        render_evidence(
            retrieved_context,
            citations,
            expanded=False
        )

        render_technical_details(
            {
                "retriever": saved_config[
                    "retriever"
                ],
                "generator": saved_config[
                    "generator"
                ],
                "prompt": saved_config[
                    "prompt"
                ],
                "retrieved_context": retrieved_context,
                "latency_ms": metadata.get(
                    "latency_ms"
                ),
                "prompt_tokens": metadata.get(
                    "prompt_tokens"
                ),
                "completion_tokens": metadata.get(
                    "completion_tokens"
                ),
            },
            saved_mode,
            saved_config,
        )


# =============================================================================
# Chat input
# =============================================================================

question = st.chat_input(
    "Ask about ATT&CK techniques, tactics, software, groups, or mitigations..."
)


if question:

    # Snapshot the exact configuration used for THIS request.
    request_mode = st.session_state.response_mode

    request_config = get_config_for_mode(
        request_mode
    )

    # -------------------------------------------------------------------------
    # Immediately display the user's question.
    # -------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -------------------------------------------------------------------------
    # Call API
    # -------------------------------------------------------------------------

    with st.chat_message("assistant"):

        status = st.status(
            "Processing your question...",
            expanded=True,
        )

        status.write(
            f"🔍 Retrieving with "
            f"**{request_config['retriever']}**..."
        )

        try:

            response = requests.post(
                f"{API_URL}/query",
                json={
                    "query": question,
                    "retriever": request_config[
                        "retriever"
                    ],
                    "generator": request_config[
                        "generator"
                    ],
                    "prompt": request_config[
                        "prompt"
                    ],
                },
                timeout=120,
            )

            response.raise_for_status()

            status.write(
                f"✍️ Generating with "
                f"**{request_config['generator']}**..."
            )

            data = response.json()

            answer = data["answer"]

            abstained = data["abstained"]

            citations = (
                data.get("citations")
                or []
            )

            retrieved_context = (
                data.get(
                    "retrieved_context"
                )
                or []
            )

            status.update(
                label="Answer ready",
                state="complete",
                expanded=False,
            )

        except requests.HTTPError as error:

            status.update(
                label="API request failed",
                state="error",
            )

            try:
                detail = response.json().get(
                    "detail",
                    str(error),
                )
            except (ValueError, AttributeError):
                detail = str(error)

            st.error(
                f"API error "
                f"({response.status_code}): "
                f"{detail}"
            )

            st.stop()

        except requests.RequestException as error:

            status.update(
                label="API unavailable",
                state="error",
            )

            st.error(
                f"Could not reach the API at "
                f"{API_URL}.\n\n"
                f"{error}"
            )

            st.stop()

        except (
            KeyError,
            ValueError,
            TypeError,
        ) as error:

            status.update(
                label="Invalid API response",
                state="error",
            )

            st.error(
                f"The API returned an unexpected "
                f"response: {error}"
            )

            st.stop()

        # ---------------------------------------------------------------------
        # Answer
        # ---------------------------------------------------------------------

        st.markdown("### 🛡️ Answer")

        if abstained:

            st.warning(
                "There was not enough supporting evidence "
                "in the retrieved knowledge to answer this "
                "question reliably."
            )

        st.markdown(answer)

        # ---------------------------------------------------------------------
        # Evidence
        # ---------------------------------------------------------------------

        render_evidence(
            retrieved_context,
            citations,
            expanded=True,
        )

        # ---------------------------------------------------------------------
        # Technical details
        # ---------------------------------------------------------------------

        render_technical_details(
            data,
            request_mode,
            request_config,
        )

    # -------------------------------------------------------------------------
    # Save result in session transcript
    # -------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "retrieved_context": retrieved_context,
            "abstained": abstained,
            "metadata": {
                "mode": request_mode,
                "retriever": data.get(
                    "retriever",
                    request_config["retriever"],
                ),
                "generator": data.get(
                    "generator",
                    request_config["generator"],
                ),
                "prompt": data.get(
                    "prompt",
                    request_config["prompt"],
                ),
                "latency_ms": data.get(
                    "latency_ms"
                ),
                "prompt_tokens": data.get(
                    "prompt_tokens"
                ),
                "completion_tokens": data.get(
                    "completion_tokens"
                ),
            },
        }
    )
import requests
import streamlit as st

API_URL = "http://localhost:8000/query"

st.set_page_config(
    page_title="MITRE ATT&CK Assistant",
    page_icon="🛡️",
    layout="wide"
)

# --------- Sidebar ---------
with st.sidebar:
    st.title("⚙️ Settings")
    retriever_type = st.selectbox(
        "Retriever",
        [
            "hybrid",
            "dense",
            "bm25",
        ]
    )

    top_k = st.slider(
        "Top K Chunks",
        min_value=1,
        max_value=20,
        value=5
    )
    st.divider()

st.title("🛡️ MITRE ATT&CK Assistant")

# --------- Chat history ---------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    chunk_id = source.get("chunk_id") or source.get("doc_id", "Unknown")
                    score = source.get("score", 0)
                    text = source.get("text", "")
                    name = source.get("name") or source.get("subject_id", "")

                    with st.expander(f"{chunk_id} | {name} | Score: {score:.3f}"):
                        st.write(text)

# --------- User input ---------
question = st.chat_input("Ask a cybersecurity question...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Call API
    with st.chat_message("assistant"):
        status = st.status("Processing query...", expanded=True)
        status.write("🔍 Retrieving relevant ATT&CK chunks...")
        try:
            response = requests.post(
                API_URL,
                json={
                    "query": question,
                    "k": top_k,
                    "retriever": retriever_type
                },
                timeout=120
            )

            response.raise_for_status()
            status.write("Generating answer...")
            data = response.json()
            answer = data.get("answer") or "\n".join([f"**Doc {r['chunk_id']}** (Score: {r['score']:.3f})\n{r['text']}" for r in data.get("results", [])])
            sources = data.get("sources") or data.get("results", [])
            status.update(label="✅ Done", state="complete")

        except Exception as e:
            status.update(label="❌ Request failed", state="error")
            st.error(str(e))
            st.stop()

        # --------- Answer ---------
        st.markdown(answer)
        if sources:
            col1, col2, col3 = st.columns(3)
            col1.metric("Retriever", retriever_type)
            col2.metric("Chunks Retrieved", len(sources))
            col3.metric("Top Score", f"{sources[0]['score']:.3f}")

        # --------------------------------------
        # --------- Sources ---------
        with st.expander("📚 Retrieved Sources"):
            for source in sources:
                chunk_id = source.get("chunk_id") or source.get("doc_id", "Unknown")
                score = source.get("score", 0)
                name = source.get("name") or source.get("subject_id", "")
                text = source.get("text", "")

                with st.expander(f"{chunk_id} | {name} | Score: {score:.3f}"):
                    st.write(text)

    # --------- Save to history ---------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )
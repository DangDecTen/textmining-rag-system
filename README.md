# Textmining RAG System

A Q&A System for MITTRE ATT&CK Knowledge Base. Ask a question, get a short answer with citations or an explicit "I don't know" if the answer isn't in the knowledge base.
- Domain: Closed-domain (only answer question inside the knowledge base)
- Question: Factoid (one way to answer), single-hop (answer located in one place of the knowledge base)
- Answer: Short answer

Our dataset is [AttackQA](https://arxiv.org/abs/2411.01073), which was created using an older version of [MITTRE ATT&CK](https://attack.mitre.org/). We used existing corpus (documents) processed by AttackQA, read the paper for details of the processing methodology.

## Table of Contents

- [System Pipeline](#system-pipeline)
- [Getting Started](#getting-started)
- [Run the System](#run-the-system)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Extending the System](#extending-the-system)
- [Where to look for what](#where-to-look-for-what)
- [References](#references)

## System Pipeline

Offline: preparing the index.

```
attackqa.parquet
        │  ingestion (src/ingestion/)
        ▼
corpus.jsonl + qa_{train,dev,test}.jsonl
        │  indexing (src/indexing/)
        ▼
Index on disk
```

Online: answering a question.

```
question + top_k
        │
        ▼
Pipeline(Retriever, Reranker, Generator)
        │
        ├─ Retriever.search()          <- bm25, dense, etc.
        ├─ Reranker.rerank()           <- cross-encoder, etc.
        ├─ Generator.generate()        <- llama, qwen, etc.
        ├─ ResponseBuilder.build()     <- turns raw text into structured answer
        │
        ▼
Answer (text, citations)
```

## Getting Started

Set up your environment.

```bash
pip install -e .            # packages and dependencies (pyproject.toml)
cp .env.example .env        # fill in secret (GROQ_API_KEY and optionally HF_TOKEN)
```

Download [AttackQA on Hugging Face](https://huggingface.co/datasets/sambanovasystems/attackqa) and parse the dataset into the Corpus for indexing, and the QA pairs with train/dev/test splits (default 80/10/10) for evaluation.

```bash
python -m src.ingestion.run_ingestion

# See src/config.py for where to add attackqa.parquet and where to get
# processed data.
```

Build the index for retrieval. This may take a while on CPU, run on CUDA for faster indexing.

```bash
python -m src.indexing.build_index

# See src/config.py for the location of the indices.
```

Set up enviroment variables for generation.

```dotenv
# Set in .env to use this generator.
GROQ_API_KEY=<YOUR_API_KEY>

# Optional, set in .env to enable faster model downloads.
HF_TOKEN=<YOUR_API_KEY>
```

You are now ready to run the system with Interactive CLI, or Web UI.

## Run the System

Run the system with default settings (`src/config.py`)

### Interactive CLI

Run the system and ask a question

```bash
python run_rag.py
python run_rag.py --list  # show every registered retriever/generator

# Q: What campaigns used attack technique ’T1562.001: Disable or Modify 
#    Tools’?
# A: The campaigns that used attack technique ’T1562.001: Disable or
#    Modify Tools’ were: ’C0002: Night Dragon’, ’C0024: SolarWinds
#    Compromise’, ’C0028: 2015 Ukraine Electric Power Attack’, ’C0029:
#    Cutting Edge’
```

### API and UI

First, open a terminal and start the API.

```bash
python -m uvicorn app.backend.api:app --reload
```

Query directly on Linux/macOS:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What campaigns used attack technique ’T1562.001: Disable or Modify Tools’?", "k": 5, "retriever": "bm25", "generator": "llama"}'
```

Query directly on Windows:

```shell
# Define a small helper
function query-rag($q) {
    irm http://127.0.0.1:8000/query -Method Post -ContentType "application/json" -Body (@{
        query=$q
        k=5
        retriever="bm25"
        generator="llama"
    } | ConvertTo-Json)
}

# Then query
query-rag "What campaigns used attack technique ’T1562.001: Disable or Modify Tools’?"
```

Next, start the frontend in a second terminal.

```bash
python -m streamlit run app/frontend/app.py
```

## Configuration

See `src/config.py` for settings of data input and output, models and model hyperparameters, default settings of the system, etc. Everything in it can be overridden via environment variables or a `.env` file — see `.env.example` for the full list.

## Models

### Retrival

The full report in evalution folder.

| retriever | n | mrr@10 | recall@1 | recall@5 | recall@10 | avg_retrieve_ms | avg_rerank_ms | avg_ms | device | rerank_candidate_k |
|:--|-:|-:|-:|-:|-:|-:|-:|-:|:--|-:|
|bm25 | 2533| 0.785| 0.694| 0.901| 0.942| 6.141| 0.000| 6.141|CPU | 0|
|bm25+cross_encoder | 2533| 0.870| 0.818| 0.938| 0.942| 2.552| 178.375| 180.927|CUDA | 20|
|dense | 2533| 0.863| 0.809| 0.932| 0.958| 228.226| 0.000| 228.226|CPU | 0|
|dense+cross_encoder | 2533| 0.882| 0.830| 0.946| 0.956| 19.151| 333.667| 352.819|CUDA | 20|
|hybrid | 2533| 0.877| 0.825| 0.946| 0.968| 107.688| 0.000| 107.688|CPU | 0|
|hybrid+cross_encoder | 2533| 0.887| 0.826| 0.967| 0.978| 24.723| 415.301| 440.024|CUDA | 20|

### Generation

Something here...









## Repository Structure

```
textmining-rag-system/
├── data/
├── src/
│   ├── config.py
│   ├── README.md
│   ├── ...
│   ├── data_models/
│   ├── ingestion/
│   │   └── README.md
│   ├── indexing/
│   │   └── README.md			# how retrieval works + how to add a retriever
│   ├── retrieval/
│   │   └── README.md           # how retrieval works + how to add a retriever
│   ├── reranking/
│   │   └── README.md           # how reranking works + how to add a reranker
│   └── generation/
│       └── README.md           # design rationale + how to add a generator
├── evaluation/
│   └── retrieval/
│   	├── main_report.md		# full evaluation and analysis on retrieval
│       └── README.md			# how to use the code
├── app/
│   ├── backend/
│   │   └── api.py            	# FastAPI
│   ├── frontend/
│   │   └── app.py            	# Streamlit chat UI
│   └── README.md
├── run_rag.py                	# interactive CLI
├── pyproject.toml            	# pip install -e .
├── requirements.txt
├── .env.example
└── README.md                 	# you are here
```

## Extending the System

Refer to `README` of each components for complete guide.

The three abstract interfaces (`Index`, `Retriever`, `Generator`) plus
three registries (`src/retrieval/registry.py`, `src/generation/registry.py`,
`src/reranking/registry.py`) are what make adding new tech mostly additive.

## References

- \[[paper](https://onlinelibrary.wiley.com/doi/full/10.1155/jece/3383674)\] Large Language Models for Security Operations Centers: A Comprehensive Survey.
- \[[paper](https://arxiv.org/abs/2411.01073)\] AttackQA: Development and Adoption of a Dataset for Assisting Cybersecurity Operations using Fine-tuned and Open-Source LLMs.

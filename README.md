# Textmining RAG System
Text Mining, build a simple RAG system

## Table of Contents

- [RAG Pipeline](#rag-pipeline)
- [Data Ingestion](#data-ingestion)
	- [Load Data](#load-data)
	- [Chunk Document](#chunk-document)
- [Indexing](#indexing)
- [Repository Structure](#repository-structure)

## RAG Pipeline

```
enterprise-attack.json (MITRE ATT&CK STIX 2.1)
        │  parse_attack.py
        ▼
attack_docs.jsonl
        │  chunk.py
        ▼
chunks.jsonl
        │  build_index.py
        ▼
faiss_index/
        │
        ▼
DenseRetriever  ──►  prompts.py  ──►  Generator
        │
        ▼
FastAPI (/query)  ──►  Streamlit chat UI
```

## Data Ingestion

Load ATT&CK data, parse into retrievable documents, do some document pre-processing, and then try different chunking techniques.

### Load Data

Details in `parse_attack.py` with tunable hyperparamerter.

|Hyperparameter|Description|
|-|-|
|MITRE_ATTACK_DOMAIN|Data from enterprise, mobile, or ICS. Feature to load data from all domains is incomplete.|
|MITRE_ATTACK_VERSION|Might use a version close to when ATT&CK was created.|
|MAX_RELATED_PER_LABEL|Tune the size of relationship context. This is to avoid heavily-used technique (e.g. T1059).|

### Chunk Document

Details of hyperparameters can be looked into corresponding source file.

|Chunking Technique|Source File|
|-|-|
|Recursive + Overlap|`chunk.py`|
|Sentence-based + Overlap|planning...|
|Late Chunking (Post-Chunking)|planning...|
|Hierarchical Chunking|planning...|

## Indexing

Empty...

## Repository Structure

```
attackqa-rag/
├── data/
│   ├── raw/                    # MITRE ATT&CK STIX/JSON dumps, AttackQA dataset
│   ├── processed/               # chunked corpus, cleaned Q&A pairs
│   └── splits/                  # train/val/test (for eval only, no fine-tuning needed)
├── src/
│   ├── ingestion/
│   │   ├── parse_attack.py      # STIX bundle -> structured docs (techniques, tactics, mitigations, groups)
│   │   └── chunk.py             # chunking strategy (see note below)
│   ├── indexing/
│   │   ├── embed.py
│   │   └── build_index.py       # vector store population
│   ├── retrieval/
│   │   ├── retriever.py         # dense / hybrid / rerank
│   │   └── hybrid.py            # BM25 + dense fusion
│   ├── generation/
│   │   ├── prompts.py
│   │   └── generate.py
│   ├── pipeline.py              # glues retriever + generator
│   └── config.py
├── eval/
│   ├── retrieval_eval.py        # Recall@k, MRR, nDCG
│   ├── generation_eval.py       # faithfulness, answer correctness, RAGAS-style or custom
│   ├── run_eval.py
│   └── results/                 # versioned eval run outputs (json/csv), one per experiment
├── notebooks/                   # exploration only
├── app/
│   ├── backend/                 # e.g. FastAPI
│   └── frontend/                # e.g. Streamlit
├── experiments/
│   └── EXPERIMENTS.md           # log: what changed, retrieval config, eval numbers — critical for stage03/04
├── tests/
├── configs/                     # yaml configs per experiment (chunk size, embedding model, k, reranker on/off)
├── requirements.txt / pyproject.toml
└── README.md
```

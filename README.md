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
attackqa.parquet
        │  run_ingestion.py
        ▼
corpus + QA pairs (train/dev/test)
        │
        ▼
Index
        │
        ▼
Retriever
```

## Data Ingestion

This includes loading the corpus for indexing, splitting QA pairs into train/dev/test for evaluation, and no chunking. AttackQA has done most of the chunking.

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

# Textmining RAG System
Text Mining, build a simple RAG system

## Table of Contents

- [RAG Pipeline](#rag-pipeline)
- [Data Ingestion](#data-ingestion)
- Indexing & Retrieval
	- [Lexical Retrieval](#lexical-retrieval)
	- [Dense Retrieval](#dense-retrieval)
- [Repository Structure](#repository-structure)
- [References](#references)

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

Parse AttackQA into a set of corpus (`corpus.jsonl`) for indexing, and QA pairs with train/dev/test splits (default is 80/10/10) for evaluation.

```bash
python -m src.ingestion.run_ingestion \
    --input data/benchmark/attackqa.parquet \
    --output-dir data/processed
```

No chunking in current implementation because AttackQA has done most of the chunking. You can explore [AttackQA dataset](https://huggingface.co/datasets/sambanovasystems/attackqa/blob/main/Getting%20Started%20with%20MITRE%20QA%20Dataset.ipynb), or look at `src/ingestion/README.md` for further information.

## Indexing & Retrieval

### Lexical Retrieval

Use [`bm25s`](https://github.com/xhluca/bm25s) library for indexing and retrieval. You can use split `dev` for hyperparamerter tunning and split `test` for final result.

```bash
# Guide
python -m src.run_bm25 --help

# Evaluation of lexical retrieval
python -m src.run_bm25 --split dev
```

The indexing stage is implemented with FAISS and a basic dense retriever.

### What it does

- Reads `data/processed/chunks.jsonl`
- Embeds each chunk with `BAAI/bge-small-en-v1.5`
- L2-normalizes the embeddings and stores them in a FAISS `IndexFlatIP`
- Writes the index and chunk metadata into `data/index/faiss_index/`

### Build the index

```bash
python -m src.indexing.build_index --chunks data/processed/chunks.jsonl --out-dir data/index/faiss_index
```

### Query the index

```python
from src.retrieval.retriever import DenseRetriever

retriever = DenseRetriever("data/index/faiss_index")
hits = retriever.search("mitigations for command and control", k=5)
```

Each hit includes the original chunk metadata plus a similarity `score`.

### Test the system

Start the API server:

```bash
uvicorn src.api:app --reload
```

Then query it:

```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"query\": \"mitigations for command and control\", \"k\": 5}"
```
## Streamlit App

### Test Retriever + Generator

```bash
python test_rag.py
```

Enter a query when prompted. 

The script will retrieve the top-5 relevant chunks and generate an answer using them.

### Run the app

Start the API server:

```bash
python -m uvicorn src.full_api:app --reload
```

Open another terminal and run the streamlit app

```bash
python -m streamlit run app.py
```
### Dense Retrieval

empty...

## Repository Structure

```
attackqa-rag/
├── data/
│   ├── benchmark/               # AttackQA dataset
│   ├── processed/               # corpus and QA pairs with train/val/test splits
│   └── index/                   # stored index for retrieval
├── src/
│   ├── data_models/             # RAG data models
│   ├── ingestion/               # parse AttackQA into corpus and QA pairs
│   ├── indexing/
│   │   ├── base.py              # abstract interface for indexing
│   │   └── bm25_index.py        # e.g. lexical retrieval
│   ├── retrieval/
│   │   ├── base.py              # abstract interface for retrieval
│   │   └── bm25_retriever.py    # e.g. lexical retrieval
│   ├── generation/
│   ├── eval/
│   │   ├── retrieval_eval.py    # calculate retrieval metrics
│   │   └── generation_eval.py   # calculate generation metrics
│   └── pipeline.py              # combine retriever, generator, and evaluation
├── eval/
│   ├── run_eval.py
│   └── results/
├── experiments/
│   └── EXPERIMENTS.md           # log: what changed, retrieval config, etc.
├── app/
│   ├── backend/                 # e.g. FastAPI
│   └── frontend/                # e.g. Streamlit
├── requirements.txt
└── README.md
```

## References

- \[[paper](https://onlinelibrary.wiley.com/doi/full/10.1155/jece/3383674)\] Large Language Models for Security Operations Centers: A Comprehensive Survey.
- \[[paper](https://arxiv.org/abs/2411.01073)\] AttackQA: Development and Adoption of a Dataset for Assisting Cybersecurity Operations using Fine-tuned and Open-Source LLMs.
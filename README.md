# Textmining RAG System
Text Mining, build a simple RAG system

## Table of Contents

- [RAG Pipeline](#rag-pipeline)
- [Data Ingestion](#data-ingestion)
- Indexing & Retrieval
	- [Lexical Retrieval](#lexical-retrieval)
	- [Dense Retrieval](#dense-retrieval)
- Generator
	- [Llama Generator](#llama-generator)
    - [Qwen Generator](#qwen-generator)
- [Streamlit App](#streamlit-app)
- [Repository Structure](#repository-structure)
- [References](#references)

## RAG Pipeline

Preparing the Index for retrieval.

```
attackqa.parquet
        │  data ingestion
        ▼
corpus + QA pairs (train/dev/test)
        │  indexing
        ▼
Index + Retriever
```

Full RAG pipeline.

```
question + top_k
        │  
        ▼
Pipeline (Retriever + Generator)
        │  
        ▼
Answer
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

Report on some experiments.

|Index & Retriever|Indexing|QA Examples|Metrics|Hyperparameters|
|-|-|-|-|-|
|`bm25s`|~15 min|2,533 (dev)|mrr: 0.724<br>recall@1: 0.639<br>recall@5: 0.831<br>recall@10: 0.886|method: lucene<br>k1: 1.5<br>b: 0.75|

### Dense Retrieval

Use [Sentence Transformers](https://sbert.net/index.html) to embed documents and questions.

```dotenv
# Optional, set a HF_TOKEN in .env to enable faster model downloads.
HF_TOKEN=<YOUR_API_KEY>
```

Use [`facebookresearch/faiss`](https://github.com/facebookresearch/faiss/wiki/) library for indexing and retrieval. You can use split `dev` for hyperparamerter tunning and split `test` for final result.

```bash
# Guide
python -m src.run_dense --help

# Evaluation of lexical retrieval
python -m src.run_dense --split dev
```

Report on some experiments.

|Index & Retriever|Indexing|QA Examples|Metrics|Hyperparameters|
|-|-|-|-|-|
|`IndexFlatIP`|~32 min|2,533 (dev)|mrr: 0.847<br>recall@1: 0.797<br>recall@5: 0.914<br>recall@10: 0.940|model: `BAAI/bge-small-en-v1.5`<br>batch: 64|

## Generator

Using `RetrievalResult` from retrieval stage as inputs, the generator is forced to output in JSON format (config with system prompt and/or API interface) which then is parsed into `GenerationResult` for a structured answer.

Test the question and number of retrieved documents (k) with your desired `Retriever` and `Generator`.

```bash
# Choose your retriever, generator, and then run the file
python -m src.test_rag
```

### Llama Generator

Use Groq API to generate with [`llama-3.3-70b-versatile`](https://console.groq.com/docs/model/llama-3.3-70b-versatile) model.

```dotenv
# Set your API key in .env to use Groq models.
GROQ_API_KEY=<YOUR_API_KEY>
```

Limits:
- Context window: 131,072
- Max output tokens: 32,768
- Latency: ~1 sec

### Qwen Generator

Download from Hugging with [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model.

```dotenv
# Optional, set a HF_TOKEN in .env to enable faster model downloads.
HF_TOKEN=<YOUR_API_KEY>
```

Limits:
- Context window: 128k
- Max output tokens: 8k
- Latency: ~15 min

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

## API (in progress...)

Start the API server:

```bash
uvicorn src.api:app --reload
```

Then query it:

```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"query\": \"mitigations for command and control\", \"k\": 5}"
```

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
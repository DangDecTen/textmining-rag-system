# Data Flow

Load chunks

```
chunks.jsonl
        ▼
load_chunks()
        ▼
List[Chunk]
```

Lexical retriever (BM25)

```
List[Chunk]
        ▼
BM25Indexer
        ▼
BM25Index
        ▼
BM25Retriever
        ▼
List[RetrievedChunk]
```

Dense retriever

```
List[Chunk]
    ▼
FAISSIndexer
    ▼
FAISSIndex
    ▼
DenseRetriever
    ▼
List[RetrievedChunk]
```

Hybrid retriever

```
List[Chunk]
        ▼
BM25Retriever + DenseRetriever
        ▼
HybridRetriever
        ▼
List[RetrievedChunk]
```
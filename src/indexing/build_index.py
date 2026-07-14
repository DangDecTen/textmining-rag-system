"""Build a FAISS dense index over the chunked ATT&CK corpus.

The index is cosine-similarity based: embeddings are L2-normalized before
being added to an inner-product FAISS index. Chunk metadata is written in the
same order as the vectors so retrieval can map result positions back to the
original chunk records.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from src.indexing.embeddings import EmbeddingBackend

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNKS_PATH = "data/processed/chunks.jsonl"
DEFAULT_INDEX_DIR = "data/index/faiss_index"
INDEX_FILE_NAME = "index.faiss"
METADATA_FILE_NAME = "chunks.jsonl"


@dataclass(frozen=True)
class IndexArtifacts:
    index_path: Path
    metadata_path: Path


def load_chunks(chunks_path: str | Path) -> list[dict]:
    with open(chunks_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def embed_texts(texts: Iterable[str], model: SentenceTransformer) -> np.ndarray:
    raise NotImplementedError


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build an index from an empty embedding matrix")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_artifacts(index: faiss.Index, chunks: list[dict], out_dir: str | Path) -> IndexArtifacts:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    index_path = out_path / INDEX_FILE_NAME
    metadata_path = out_path / METADATA_FILE_NAME

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return IndexArtifacts(index_path=index_path, metadata_path=metadata_path)


def build_index(chunks_path: str | Path = DEFAULT_CHUNKS_PATH, out_dir: str | Path = DEFAULT_INDEX_DIR,
                model_name: str = EMBED_MODEL_NAME) -> IndexArtifacts:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    model = EmbeddingBackend(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    index = build_faiss_index(embeddings)
    if not model.using_sentence_transformers:
        print("sentence-transformers not available; using hashed fallback embeddings")
    return save_artifacts(index, chunks, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS dense index for ATT&CK chunks")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH, help="Path to the chunk JSONL file")
    parser.add_argument("--out-dir", default=DEFAULT_INDEX_DIR, help="Directory to write the FAISS artifacts")
    parser.add_argument("--model", default=EMBED_MODEL_NAME, help="SentenceTransformer model name")
    args = parser.parse_args()

    artifacts = build_index(args.chunks, args.out_dir, args.model)
    print(f"Saved FAISS index to {artifacts.index_path}")
    print(f"Saved chunk metadata to {artifacts.metadata_path}")


if __name__ == "__main__":
    main()

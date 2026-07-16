import json
from pathlib import Path

from src.data_models.chunk import Chunk


def load_chunks(path: str | Path) -> list[Chunk]:
    """
    Load ATT&CK chunks from a JSONL file.
    """

    path = Path(path)

    chunks: list[Chunk] = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            obj = json.loads(line)

            chunks.append(
                Chunk(
                    chunk_id=obj["chunk_id"],
                    doc_id=obj["doc_id"],
                    attack_domain=obj["attack_domain"],
                    attack_type=obj["attack_type"],
                    name=obj["name"],
                    text=obj["text"],
                    url=obj["url"],
                )
            )

    return chunks
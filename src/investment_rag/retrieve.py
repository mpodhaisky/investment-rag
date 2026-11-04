import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    embeddings = np.load(DATA_DIR / "embeddings.npy")

    with (DATA_DIR / "chunks.jsonl").open() as f:
        documents = [json.loads(line) for line in f]

    assert len(documents) == len(embeddings)

    model = SentenceTransformer(MODEL)

    while True:
        query = input("\nQuestion (q to quit): ").strip()

        if query.lower() == "q":
            break
        if not query:
            continue

        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
        )

        # Cosine similarity (vectors already normalized)
        scores = embeddings @ query_embedding

        # Retrieve top 20
        k = min(20, len(scores))
        indices = np.argsort(scores)[-k:][::-1]

        for i in indices:
            print(f"\nScore: {scores[i]:.4f}")
            print(documents[i]["text"][:500])


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from .rerank import MODEL as RERANK_MODEL, rerank

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    embeddings = np.load(DATA_DIR / "embeddings.npy")

    with (DATA_DIR / "chunks.jsonl").open() as f:
        documents = [json.loads(line) for line in f]

    assert len(documents) == len(embeddings)

    model = SentenceTransformer(MODEL)
    reranker = CrossEncoder(RERANK_MODEL)

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

        # Cosine similarity
        scores = embeddings @ query_embedding

        # Retrieve top 20
        k = min(20, len(scores))
        indices = np.argsort(scores)[-k:][::-1]

        # Rerank and keep top 5
        candidates = [documents[i] for i in indices]
        results = rerank(query, candidates, reranker)

        for doc in results:
            print(f"\nRerank score: {doc['rerank_score']:.4f}")
            print(doc["text"][:500])


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    with (DATA_DIR / "chunks.jsonl").open() as f:
        texts = [json.loads(line)["text"] for line in f]

    model = SentenceTransformer(MODEL)

    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(DATA_DIR / "embeddings.npy", embeddings)
    print(f"Saved embeddings: {embeddings.shape}")


if __name__ == "__main__":
    main()

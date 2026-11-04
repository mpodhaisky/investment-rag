from sentence_transformers import CrossEncoder

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def rerank(query, documents, model, top_k=5):
    pairs = [(query, doc["text"]) for doc in documents]

    scores = model.predict(pairs)

    results = [
        {**doc, "rerank_score": float(score)}
        for doc, score in zip(documents, scores)
    ]

    return sorted(
        results,
        key=lambda doc: doc["rerank_score"],
        reverse=True,
    )[:top_k]

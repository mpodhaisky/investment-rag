from openai import OpenAI

MODEL = "gpt-6-luna"


def generate(query, documents):
    client = OpenAI()

    context = "\n\n".join(
        f"[{i}] {doc['text']}"
        for i, doc in enumerate(documents, 1)
    )

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "none"},
        instructions=(
            "You are a personal finance assistant. "
            "Answer using only the provided context. "
            "If the context is insufficient, say so. "
            "Cite sources using [1], [2], etc."
        ),
        input=f"Context:\n{context}\n\nQuestion: {query}",
        max_output_tokens=600,
    )

    return response.output_text

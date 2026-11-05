# Investment RAG

A small, educational command-line retrieval-augmented generation (RAG) project in Python. It uses NumPy for vector search, local embedding and reranking models, and an OpenAI API model for answer generation—without LangChain or a vector database.

## Pipeline

```text
Hugging Face QA data (JSONL)
  -> keyword-filter finance questions (filtered.jsonl)
  -> chunk answers (140 words, 25-word overlap)
  -> all-MiniLM-L6-v2 embeddings
  -> NumPy embeddings.npy

User question
  -> embedding -> cosine similarity -> Top 20
  -> cross-encoder reranking -> Top 5
  -> OpenAI LLM -> answer + displayed retrieved sources
```

| Stage | Model / implementation |
| --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| Similarity | NumPy dot product over normalized vectors |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Generation | OpenAI Responses API (model set in `src/investment_rag/generate.py`) |

Embeddings and reranking run locally. Only answer generation uses the OpenAI API.

## 1. Install uv and dependencies

Requirements: macOS, Linux, or Windows; Python 3.9+; an internet connection for downloads; and an OpenAI API key for generation.

On macOS:

```bash
brew install uv
```

Then clone and set up the existing uv project:

```bash
git clone https://github.com/mpodhaisky/investment-rag.git
cd investment-rag
uv sync
```

`uv sync` installs dependencies from `pyproject.toml`/`uv.lock` and creates `.venv` as needed. **Do not run `uv init` in an existing clone**; this repository is already initialized.

## 2. Download the dataset

Dataset: [Gamestatue/adaption-personal-finance-qa](https://huggingface.co/datasets/Gamestatue/adaption-personal-finance-qa) (Hugging Face).

From the project root:

```bash
mkdir -p data
curl -L --fail \
  'https://huggingface.co/datasets/Gamestatue/adaption-personal-finance-qa/resolve/main/data/train_0.jsonl' \
  -o data/raw.jsonl
```

The pipeline reads the dataset's `input` (question) and `output` (answer) fields. Despite its name, the dataset includes unrelated topics. A simple **keyword filter** is applied before chunking to reduce out-of-domain records.

## 3. Filter the dataset

```bash
uv run python -m investment_rag.filter
```

This runs `src/investment_rag/filter.py`, reads `data/raw.jsonl`, and creates `data/filtered.jsonl`. It keeps a record if its **question** (`input`) contains at least one finance-related keyword such as `budget`, `investment`, `etf`, `tax`, `loan`, or `retirement`. Matching is case-insensitive and based on whole words; the answer is **not** considered during filtering. The filter prints how many records it retained.

This is a deliberately simple baseline, **not a trained finance classifier**. It may exclude relevant questions that use other wording and retain unrelated questions containing a finance keyword. Edit `KEYWORDS` in `filter.py` to adjust the scope.

## 4. Chunk the filtered data

```bash
uv run python -m investment_rag.chunk
```

This runs `src/investment_rag/chunk.py`, reads `data/filtered.jsonl`, and writes `data/chunks.jsonl`. Run filtering first.

- Each nonempty QA record becomes one or more chunks.
- An answer is split into up to **140 words**, with **25 words** of overlap between chunks.
- Each chunk also includes its original question.
- IDs look like `123:0` (row number **within `filtered.jsonl`** and chunk index).

This is word-based rather than tokenizer-based chunking, so exceptionally long prompts or chunks may be truncated by the embedding model.

## 5. Generate the NumPy embedding matrix

```bash
uv run python -m investment_rag.embed
```

This loads `data/chunks.jsonl`, embeds every chunk using `all-MiniLM-L6-v2`, normalizes the resulting vectors, and saves them as `data/embeddings.npy`.

**Keep the files aligned:** row `i` of `embeddings.npy` corresponds to line `i` of `chunks.jsonl`. If you change the keywords or filtering logic, rerun **filter → chunk → embed** in that order. If you change the chunks, run embedding again.

## 6. Configure OpenAI

Create a `.env` file at the project root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The API generation model is configured in `src/investment_rag/generate.py` via its `MODEL` constant; change it to a model available to your account if necessary. API requests consume paid API credits. `.env` and `data/` are ignored by Git; never commit API keys or sensitive personal finances.

## 7. Run the CLI

```bash
uv run --env-file .env python -m investment_rag.retrieve
```

Example interaction:

```text
Question (q to quit): What is an ETF?

Answer:
... [1]

--- Retrieved Sources ---
[1] Source: 123:0
Rerank score: ...
Question: ...
Answer: ...
```

Enter `q` to quit.

For each question, the CLI uses the same embedding model, computes cosine similarity via a NumPy dot product, keeps the top **20** candidates, reranks them with a cross-encoder, selects the top **5**, and passes those chunks to the language model. The numbered sources displayed below the answer correspond to retrieved dataset chunks, **not externally verified financial references**. Citations generated by the LLM may be wrong.

## Project layout

```text
investment-rag/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env                    # create locally; not committed
├── data/                   # create locally; not committed
│   ├── raw.jsonl
│   ├── filtered.jsonl
│   ├── chunks.jsonl
│   └── embeddings.npy
└── src/investment_rag/
    ├── __init__.py
    ├── filter.py
    ├── chunk.py
    ├── embed.py
    ├── retrieve.py
    ├── rerank.py
    └── generate.py
```

## Limitations

This is a learning project, **not financial advice**. It does not ingest personal bank records or perform financial calculations. It uses brute-force retrieval instead of an approximate nearest-neighbor index, has no automated relevance or answer-quality evaluation, and uses only a basic question-keyword filter rather than reliable domain classification. Answers may be incorrect; verify decisions independently.

## License and third-party data

The repository's original source code is licensed under the [MIT License](LICENSE). The dataset and pretrained models are separate third-party resources governed by their respective licenses and terms. This project's license does **not** relicense those resources; review their original pages before redistributing them.

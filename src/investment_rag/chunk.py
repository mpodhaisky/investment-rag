
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CHUNK_SIZE = 140  # Words per chunk
OVERLAP = 25      # Overlapping words


def chunk_text(text):
    words = text.split()
    step = CHUNK_SIZE - OVERLAP

    for start in range(0, len(words), step):
        yield " ".join(words[start:start + CHUNK_SIZE])

        if start + CHUNK_SIZE >= len(words):
            break


def main():
    count = 0

    with (DATA_DIR / "filtered.jsonl").open(encoding="utf-8") as source, \
         (DATA_DIR / "chunks.jsonl").open("w", encoding="utf-8") as target:

        for row_id, line in enumerate(source):
            record = json.loads(line)

            question = (record.get("input") or "").strip()
            answer = (record.get("output") or "").strip()

            if not question or not answer:
                continue

            for chunk_id, chunk in enumerate(chunk_text(answer)):
                document = {
                    "id": f"{row_id}:{chunk_id}",
                    "text": f"Question: {question}\nAnswer: {chunk}",
                }

                target.write(json.dumps(document, ensure_ascii=False) + "\n")
                count += 1

    print(f"Created {count} chunks")


if __name__ == "__main__":
    main()

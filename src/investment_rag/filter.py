
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

KEYWORDS = {
    "money", "finance", "financial", "budget",
    "savings", "saving", "invest", "investment",
    "stocks", "stock", "etf", "bonds", "portfolio",
    "tax", "taxes", "income", "salary", "debt",
    "loan", "mortgage", "credit", "bank", "banking",
    "interest", "inflation", "retirement", "pension",
    "insurance", "expenses", "crypto"
}


def relevant(question):
    words = set(re.findall(r"[a-z]+", question.lower()))
    return bool(words & KEYWORDS)


def main():
    count = 0

    with (DATA_DIR / "raw.jsonl").open() as source, \
         (DATA_DIR / "filtered.jsonl").open("w") as target:

        for line in source:
            record = json.loads(line)
            question = record.get("input") or ""

            if isinstance(question, str) and relevant(question):
                target.write(line)
                count += 1

    print(f"Kept {count} records")


if __name__ == "__main__":
    main()


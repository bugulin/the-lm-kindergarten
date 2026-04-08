import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

"""
What it does:
- reads the JSON file
- for each object, takes syllogism
- computes sha256(syllogism)
- overwrites id
- saves back into the same file
"""

"""
Usage: 
Python hashing.py path/to/data_file.json
"""

def hash_syllogism(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def load_json(path: Path )-> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")

def overwrite_ids_with_hashes(path: Path) -> None:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError("Expected the JSON file to contain a list of objects.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not a JSON object.")

        if "syllogism" not in item:
            raise ValueError(f'Item at index {i} is missing the "syllogism" field.')

        syllogism = item["syllogism"]
        if not isinstance(syllogism, str):
            raise ValueError(f'"syllogism" at index {i} must be a string.')

        item["id"] = hash_syllogism(syllogism)

    save_json(path, data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hash each "syllogism" in a JSON file and overwrite the corresponding "id".'
    )
    parser.add_argument("file", help="Path to the JSON file to process.")
    args = parser.parse_args()

    path = Path(args.file)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    overwrite_ids_with_hashes(path)
    print(f"Updated IDs in {path}")


if __name__ == "__main__":
    main()
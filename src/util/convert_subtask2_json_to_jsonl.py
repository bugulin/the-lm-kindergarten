import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

input_file = ROOT_DIR / "data" / "2" / "data.json"
output_file = ROOT_DIR / "data" / "2" / "data.jsonl"

with input_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

with output_file.open("w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Converted {len(data)} records.")
print(f"Saved to: {output_file}")
import json
import random
from pathlib import Path

random.seed(42)

input_path = Path("data/2/data.json")
train_path = Path("data/2/train_data.json")
valid_path = Path("data/2/valid_data.json")

with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

random.shuffle(data)

valid_size = int(len(data) * 0.15)
valid_data = data[:valid_size]
train_data = data[valid_size:]

with train_path.open("w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with valid_path.open("w", encoding="utf-8") as f:
    json.dump(valid_data, f, indent=2, ensure_ascii=False)

print(f"Train: {len(train_data)}")
print(f"Valid: {len(valid_data)}")
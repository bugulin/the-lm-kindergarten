import json
import argparse
from pathlib import Path


def merge_json_files(input_files, output_file):
    """
    Merge multiple JSON files into one JSON file.

    Assumes each input JSON file contains a list of samples.
    """

    merged_data = []

    for file_path in input_files:
        path = Path(file_path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected {file_path} to contain a list of samples.")

        merged_data.extend(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Merged {len(input_files)} files.")
    print(f"Total samples: {len(merged_data)}")
    print(f"Saved merged file to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple JSON dataset files into one JSON file."
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="JSON files to merge"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="merged.json",
        help="Output JSON file name"
    )

    args = parser.parse_args()

    merge_json_files(
        input_files=args.input_files,
        output_file=args.output
    )
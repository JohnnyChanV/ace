#!/usr/bin/env python3
"""
Prepare essay comment data for ACE framework.

Converts the original JSON format to ACE-compatible JSONL format,
and splits dev_data.json into train/val sets.

Usage:
    python -m eval.essay_comment.prepare_data
"""
import os
import json
import random
import argparse

# Paths relative to ace/ root
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "essay_comment")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


LABEL_MAP = {
    0: "Without Explanation",
    1: "With Explanation",
}


def convert_to_ace_format(samples):
    """
    Convert raw essay comment samples to ACE's standardized JSONL format.
    
    ACE expects: {"context": str, "question": str, "target": str}
    Targets use semantic labels: "With Explanation" / "Without Explanation"
    """
    converted = []
    for item in samples:
        comment = item.get("input", item.get("Comment", ""))
        label = int(item.get("label", item.get("Explanation (human code)", -1)))
        dimension = item.get("Dimension.Name", "unknown")

        converted.append({
            "context": comment.strip(),
            "target": LABEL_MAP[label],
            "others": {
                "dimension": dimension,
                "com_no": item.get("Com.No", -1),
            }
        })
    return converted


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare essay comment data for ACE")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of dev_data to use as validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--test_file", type=str, default="test_data_1k.json",
                        help="Which test file to use (default: test_data_1k.json)")
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Load raw data ---
    dev_path = os.path.join(RAW_DATA_DIR, "dev_data.json")
    test_path = os.path.join(RAW_DATA_DIR, args.test_file)

    print(f"Loading dev data from {dev_path}")
    dev_data = load_json(dev_path)
    print(f"  -> {len(dev_data)} samples")

    print(f"Loading test data from {test_path}")
    test_data = load_json(test_path)
    print(f"  -> {len(test_data)} samples")

    # --- Split dev into train / val ---
    indices = list(range(len(dev_data)))
    random.shuffle(indices)
    val_size = int(len(dev_data) * args.val_ratio)
    val_indices = set(indices[:val_size])

    train_raw = [dev_data[i] for i in range(len(dev_data)) if i not in val_indices]
    val_raw = [dev_data[i] for i in range(len(dev_data)) if i in val_indices]

    print(f"\nSplit: train={len(train_raw)}, val={len(val_raw)}")

    # --- Convert to ACE format ---
    train_ace = convert_to_ace_format(train_raw)
    val_ace = convert_to_ace_format(val_raw)
    test_ace = convert_to_ace_format(test_data)

    # Print label distributions
    for name, data in [("train", train_ace), ("val", val_ace), ("test", test_ace)]:
        labels = [d["target"] for d in data]
        wo = labels.count("Without Explanation")
        w = labels.count("With Explanation")
        print(f"  {name}: total={len(data)}, Without Explanation={wo}, With Explanation={w}")

    # --- Save JSONL files ---
    save_jsonl(train_ace, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(val_ace, os.path.join(OUTPUT_DIR, "val.jsonl"))
    save_jsonl(test_ace, os.path.join(OUTPUT_DIR, "test.jsonl"))

    # --- Save config ---
    config = {
        "essay_comment": {
            "train_data": "./eval/essay_comment/data/train.jsonl",
            "val_data": "./eval/essay_comment/data/val.jsonl",
            "test_data": "./eval/essay_comment/data/test.jsonl"
        }
    }
    config_path = os.path.join(OUTPUT_DIR, "task_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved task config to {config_path}")


if __name__ == "__main__":
    main()

"""
Data processor for essay comment explanation classification task.

Task: Binary classification — determine whether a writing feedback comment
includes an explanation ("With Explanation") or not ("Without Explanation").
"""

import os
import json
from typing import List, Dict, Any
from utils import extract_answer

# ─── Semantic label constants ────────────────────────────────────────────────
LABEL_POS = "With Explanation"
LABEL_NEG = "Without Explanation"

# ─── Task description (used as the "question" in ACE's standard format) ──────
# Derived from the annotation codebook for this task.
TASK_QUESTION = (
    "You are given a feedback comment on a piece of student writing. "
    "Your task is to determine whether the comment contains an Explanation.\n\n"
    "Definition: A comment is coded as having an Explanation if it provides "
    "a rationale or justification. The text must go beyond a simple statement "
    "by providing specific details or demonstrating causal thinking "
    "(cause and consequence).\n\n"
    "Coding Criteria — check whether at least one sentence in the comment:\n"
    "  - Explains why something in the writing is problematic.\n"
    "  - Explains why something is praiseworthy.\n"
    "  - Explains the reasoning behind a suggestion.\n\n"
    "If ANY of the above criteria is met, answer: \"With Explanation\"\n"
    "If NONE of the above criteria is met, answer: \"Without Explanation\"\n\n"
    "You MUST answer with EXACTLY one of the two labels above."
)


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


class DataProcessor:
    """
    Processor for the essay comment explanation classification task.

    Implements the three methods required by the ACE framework:
      - process_task_data()
      - answer_is_correct()
      - evaluate_accuracy()
    """

    def __init__(self, task_name: str = "essay_comment"):
        self.task_name = task_name

    # ── 1. Convert raw JSONL dicts to ACE's standardized format ──────────────
    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Convert raw data to ACE's standard format:
            {"context": str, "question": str, "target": str, "others": dict}

        The raw JSONL (produced by prepare_data.py) has:
            context  – the comment text
            target   – "With Explanation" or "Without Explanation"
            others   – metadata (dimension, com_no, …)
        """
        processed = []
        for item in raw_data:
            processed.append({
                "context": item.get("context", ""),
                "question": TASK_QUESTION,
                "target": str(item.get("target", "")),
                "others": item.get("others", {}),
            })
        return processed

    # ── 2. Single-sample correctness check ───────────────────────────────────
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check whether the predicted label matches the ground truth.
        Normalises both sides before comparison.
        """
        pred_label = self._normalize_label(predicted)
        gt_label = self._normalize_label(ground_truth)
        return pred_label == gt_label

    # ── 3. Batch accuracy evaluation ─────────────────────────────────────────
    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        """
        Compute overall accuracy and print a brief classification report.
        Positive class = "With Explanation"
        """
        if len(out) != len(target):
            raise ValueError("Prediction and target lists must have the same length.")

        correct = 0
        tp = fp = fn = tn = 0
        for pred_raw, gt_raw in zip(out, target):
            pred = self._normalize_label(pred_raw)
            gt = self._normalize_label(gt_raw)

            if pred == gt:
                correct += 1

            # Confusion matrix (positive = With Explanation)
            if gt == LABEL_POS and pred == LABEL_POS:
                tp += 1
            elif gt == LABEL_NEG and pred == LABEL_POS:
                fp += 1
            elif gt == LABEL_POS and pred == LABEL_NEG:
                fn += 1
            else:
                tn += 1

        total = len(out)
        accuracy = correct / total if total > 0 else 0.0

        # Macro-F1
        precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_1 = (2 * precision_1 * recall_1 / (precision_1 + recall_1)
                if (precision_1 + recall_1) > 0 else 0.0)

        precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_0 = (2 * precision_0 * recall_0 / (precision_0 + recall_0)
                if (precision_0 + recall_0) > 0 else 0.0)

        macro_f1 = (f1_0 + f1_1) / 2

        # Cohen's Kappa
        pe = (((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp))
              / (total * total)) if total > 0 else 0
        kappa = (accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

        print(f"\n{'─'*55}")
        print(f"  Accuracy  : {accuracy:.4f}  ({correct}/{total})")
        print(f"  Macro-F1  : {macro_f1:.4f}")
        print(f"  Cohen's κ : {kappa:.4f}")
        print(f"  With Explanation     P={precision_1:.3f}  R={recall_1:.3f}  F1={f1_1:.3f}")
        print(f"  Without Explanation  P={precision_0:.3f}  R={recall_0:.3f}  F1={f1_0:.3f}")
        print(f"  Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"{'─'*55}\n")

        return accuracy

    # ── Internal helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _normalize_label(text: str) -> str:
        """
        Robustly convert a model response (or ground truth) to one of the
        two canonical labels:
            "With Explanation"  /  "Without Explanation"

        Handles:
          - Exact semantic labels
          - Plain digits: "0", "1"
          - Yes/No style
          - Keyword-based heuristics
          - JSON-style final_answer from ACE generator
        """
        if text is None:
            return "UNKNOWN"

        text = text.strip()

        # ── Exact match on canonical labels ──────────────────────────────
        if text == LABEL_POS or text == LABEL_NEG:
            return text

        text_lower = text.lower()

        # ── Direct digit ─────────────────────────────────────────────────
        if text == "1":
            return LABEL_POS
        if text == "0":
            return LABEL_NEG

        # ── Semantic keyword matching ────────────────────────────────────
        # Check negative FIRST (many contain substring of positive)
        negative_keywords = [
            "without explanation",
            "no explanation",
            "does not include an explanation",
            "does not provide an explanation",
            "does not contain an explanation",
            "has no explanation",
            "lacks explanation",
            "no explanatory",
            "not include an explanation",
            "not provide an explanation",
            "none of the above criteria",
        ]
        positive_keywords = [
            "with explanation",
            "includes an explanation",
            "provides an explanation",
            "contains explanatory",
            "contains an explanation",
            "has an explanation",
            "has explanation",
            "explanation is present",
            "there is an explanation",
            "there is at least an explanation",
            "explain problem",
            "explain praise",
            "explain suggestion",
        ]

        for kw in negative_keywords:
            if kw in text_lower:
                return LABEL_NEG
        for kw in positive_keywords:
            if kw in text_lower:
                return LABEL_POS

        # ── Yes / No ────────────────────────────────────────────────────
        if text_lower.startswith("yes"):
            return LABEL_POS
        if text_lower.startswith("no"):
            return LABEL_NEG

        # ── Try extracting from JSON-style answer ───────────────────────
        try:
            extracted = extract_answer(text)
            if extracted and extracted != "No final answer found":
                return DataProcessor._normalize_label(extracted)
        except Exception:
            pass

        # ── Last resort: search for a bare digit ────────────────────────
        for ch in text:
            if ch == "1":
                return LABEL_POS
            if ch == "0":
                return LABEL_NEG

        return "UNKNOWN"  # Unable to parse

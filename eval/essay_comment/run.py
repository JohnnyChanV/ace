#!/usr/bin/env python3
"""
ACE runner for the essay comment explanation classification task.

Usage examples:
    # ── Local vLLM (load-balanced across ports 8000-8004) ─────────────
    python -m eval.essay_comment.run \
        --task_name essay_comment \
        --mode offline \
        --api_provider local \
        --local_ports 8000,8001,8002,8003,8004 \
        --generator_model Qwen/Qwen3-4B-Thinking-2507 \
        --reflector_model Qwen/Qwen3-4B-Thinking-2507 \
        --curator_model Qwen/Qwen3-4B-Thinking-2507 \
        --save_path ./results/essay_comment

    # ── Online mode ───────────────────────────────────────────────────
    python -m eval.essay_comment.run \
        --task_name essay_comment \
        --mode online \
        --api_provider local \
        --save_path ./results/essay_comment_online

    # ── Eval-only mode (with a pre-trained playbook) ──────────────────
    python -m eval.essay_comment.run \
        --task_name essay_comment \
        --mode eval_only \
        --api_provider local \
        --initial_playbook_path ./results/essay_comment/final_playbook.txt \
        --save_path ./results/essay_comment_eval
"""
import os
import json
import argparse
from datetime import datetime
from .data_processor import DataProcessor, load_data

from ace import ACE
from utils import initialize_clients


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ACE System — Essay Comment Explanation Classification"
    )

    # ── Task configuration ───────────────────────────────────────────────────
    parser.add_argument("--task_name", type=str, default="essay_comment",
                        help="Task name (default: essay_comment)")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Path to an initial playbook file (optional)")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode")

    # ── Model configuration ──────────────────────────────────────────────────
    parser.add_argument("--api_provider", type=str, default="local",
                        choices=["sambanova", "together", "openai", "local"],
                        help="API provider ('local' for vLLM on localhost)")
    parser.add_argument("--local_ports", type=str, default="8000,8001,8002,8003",
                        help="Comma-separated ports for local vLLM load balancing "
                             "(default: 8000,8001,8002,8003)")
    parser.add_argument("--generator_model", type=str,
                        default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Model for the generator agent")
    parser.add_argument("--reflector_model", type=str,
                        default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Model for the reflector agent")
    parser.add_argument("--curator_model", type=str,
                        default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Model for the curator agent")

    # ── Training configuration ───────────────────────────────────────────────
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_num_rounds", type=int, default=3,
                        help="Max reflection rounds for incorrect answers")
    parser.add_argument("--curator_frequency", type=int, default=1,
                        help="Run curator every N training steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate on val/test every N training steps")
    parser.add_argument("--online_eval_frequency", type=int, default=15,
                        help="Window size for online mode evaluation")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save intermediate playbooks every N steps")

    # ── System configuration ─────────────────────────────────────────────────
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max tokens for LLM responses (binary cls needs fewer)")
    parser.add_argument("--playbook_token_budget", type=int, default=40000,
                        help="Token budget for playbook")
    parser.add_argument("--test_workers", type=int, default=20,
                        help="Parallel workers for test evaluation")

    # ── Prompt / output configuration ────────────────────────────────────────
    parser.add_argument("--json_mode", action="store_true",
                        help="Enable JSON mode for LLM calls")
    parser.add_argument("--no_ground_truth", action="store_true",
                        help="Do NOT use ground truth in reflection")
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true",
                        help="Enable bulletpoint deduplication / merging")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90,
                        help="Similarity threshold for bulletpoint analyzer")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save all results")

    return parser.parse_args()


def preprocess_data(task_name, data_config, mode):
    """
    Load and preprocess data splits.

    Returns:
        (train_samples, val_samples, test_samples, data_processor)
    """
    processor = DataProcessor(task_name=task_name)

    if mode in ("online", "eval_only"):
        train_samples = None
        val_samples = None
        if "test_data" not in data_config:
            raise ValueError(f"{mode} mode requires 'test_data' in config.")
        test_samples = processor.process_task_data(load_data(data_config["test_data"]))
        print(f"{mode.replace('_', ' ').title()} mode: {len(test_samples)} test samples")
    else:
        train_samples = processor.process_task_data(load_data(data_config["train_data"]))
        val_samples = processor.process_task_data(load_data(data_config["val_data"]))
        test_samples = (
            processor.process_task_data(load_data(data_config["test_data"]))
            if "test_data" in data_config else []
        )
        print(f"Offline mode: train={len(train_samples)}, val={len(val_samples)}, "
              f"test={len(test_samples)}")

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


def main():
    args = parse_args()

    # ── Parse local ports ──────────────────────────────────────────────────
    local_ports = None
    if args.api_provider == "local":
        local_ports = [int(p.strip()) for p in args.local_ports.split(",") if p.strip()]

    print(f"\n{'='*60}")
    print(f"ACE SYSTEM — Essay Comment Explanation Classification")
    print(f"{'='*60}")
    print(f"Task      : {args.task_name}")
    print(f"Mode      : {args.mode.upper().replace('_', ' ')}")
    print(f"Generator : {args.generator_model}")
    print(f"Reflector : {args.reflector_model}")
    print(f"Curator   : {args.curator_model}")
    print(f"Provider  : {args.api_provider}")
    if local_ports:
        print(f"Ports     : {local_ports}")
    print(f"{'='*60}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    config_path = os.path.join(os.path.dirname(__file__), "data", "task_config.json")
    with open(config_path, "r") as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name, task_config[args.task_name], args.mode
    )

    # ── Load initial playbook ────────────────────────────────────────────────
    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Starting with an empty playbook\n")

    # ── Create ACE system ────────────────────────────────────────────────────
    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
        local_ports=local_ports,
    )

    # ── Build run config ─────────────────────────────────────────────────────
    config = {
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "online_eval_frequency": args.online_eval_frequency,
        "save_steps": args.save_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "task_name": args.task_name,
        "mode": args.mode,
        "json_mode": args.json_mode,
        "no_ground_truth": args.no_ground_truth,
        "save_dir": args.save_path,
        "test_workers": args.test_workers,
        "initial_playbook_path": args.initial_playbook_path,
        "use_bulletpoint_analyzer": args.use_bulletpoint_analyzer,
        "bulletpoint_analyzer_threshold": args.bulletpoint_analyzer_threshold,
        "api_provider": args.api_provider,
    }

    # ── Run ──────────────────────────────────────────────────────────────────
    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config,
    )

    print("\nDone! Results saved to:", args.save_path)


if __name__ == "__main__":
    main()

"""
SetFit Implementation: Efficient Few-Shot Learning Without Prompts
Paper: Tunstall et al. (2022) - arXiv:2209.11055

This script reproduces the SetFit framework using the official HuggingFace SetFit library,
running experiments on Emotion and AG News datasets with N=8 and N=64 samples per class.
"""

import json
import time
import warnings
from datetime import datetime

import numpy as np
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


# ─── Configuration ───────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "dataset": "emotion",
        "text_col": "text",
        "label_col": "label",
        "num_classes": 6,
        "metric": "accuracy",
        "description": "Emotion classification (6 classes: anger, fear, joy, love, sadness, surprise)",
    },
    {
        "dataset": "ag_news",
        "text_col": "text",
        "label_col": "label",
        "num_classes": 4,
        "metric": "accuracy",
        "description": "AG News topic classification (4 classes: World, Sports, Business, Sci/Tech)",
    },
]

SAMPLE_SIZES = [8, 64]  # N samples per class, matching the paper

MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"  # SetFit_MPNet from the paper

SEED = 42
NUM_EPOCHS = 1  # Paper uses 1 epoch for ST fine-tuning
BATCH_SIZE = 16
NUM_ITERATIONS = 20  # R=20 pairs per class (paper default)


# ─── Helper functions ─────────────────────────────────────────────────────────

def sample_few_shot(dataset, label_col, n_per_class, seed=42):
    """Sample n examples per class (few-shot setup)."""
    from collections import defaultdict
    import random
    random.seed(seed)

    by_class = defaultdict(list)
    for i, example in enumerate(dataset):
        by_class[example[label_col]].append(i)

    selected_indices = []
    for label, indices in by_class.items():
        chosen = random.sample(indices, min(n_per_class, len(indices)))
        selected_indices.extend(chosen)

    return dataset.select(selected_indices)


def run_setfit_experiment(config, n_per_class, seed=SEED):
    """Run a single SetFit experiment and return metrics."""
    print(f"\n{'='*60}")
    print(f"Dataset: {config['dataset'].upper()} | N={n_per_class} per class")
    print(f"{'='*60}")

    # Load dataset
    print("Loading dataset...")
    if config["dataset"] == "emotion":
        raw = load_dataset("dair-ai/emotion", trust_remote_code=True)
        train_data = raw["train"]
        test_data = raw["test"]
    elif config["dataset"] == "ag_news":
        raw = load_dataset("ag_news")
        train_data = raw["train"]
        test_data = raw["test"]
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    # Sample few-shot training data
    few_shot_train = sample_few_shot(
        train_data, config["label_col"], n_per_class, seed=seed
    )
    print(f"Few-shot train size: {len(few_shot_train)} ({n_per_class} per class)")
    print(f"Test size: {len(test_data)}")

    # Load SetFit model
    print(f"Loading model: {MODEL_NAME}")
    model = SetFitModel.from_pretrained(
        MODEL_NAME,
        labels=list(range(config["num_classes"])),
    )

    # Training arguments (matching paper: cosine loss, 1 epoch, batch=16, R=20)
    args = TrainingArguments(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        num_iterations=NUM_ITERATIONS,
        seed=seed,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=few_shot_train,
        column_mapping={
            config["text_col"]: "text",
            config["label_col"]: "label",
        },
    )

    # Train
    print("Fine-tuning Sentence Transformer + training classifier head...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    # Evaluate
    print("Evaluating on test set...")
    preds = model.predict(test_data[config["text_col"]])
    labels = test_data[config["label_col"]]

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True)

    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    return {
        "accuracy": float(acc),
        "train_time_seconds": float(train_time),
        "train_size": len(few_shot_train),
        "test_size": len(test_data),
        "classification_report": report,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = {
        "paper": "SetFit: Efficient Few-Shot Learning Without Prompts (arXiv:2209.11055)",
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }

    for config in EXPERIMENTS:
        dataset_name = config["dataset"]
        results["experiments"][dataset_name] = {
            "description": config["description"],
            "results_by_n": {},
        }

        for n in SAMPLE_SIZES:
            try:
                metrics = run_setfit_experiment(config, n_per_class=n)
                results["experiments"][dataset_name]["results_by_n"][str(n)] = metrics
                print(f"\n✓ {dataset_name} N={n}: Accuracy = {metrics['accuracy']*100:.2f}%")
            except Exception as e:
                print(f"\n✗ {dataset_name} N={n}: Error - {e}")
                results["experiments"][dataset_name]["results_by_n"][str(n)] = {
                    "error": str(e)
                }

    # Save results
    output_path = "results/setfit_results.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to {output_path}")
    print_summary(results)


def print_summary(results):
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"Model: {results['model']}")
    print()

    paper_results = {
        "emotion": {8: 48.8, 64: 76.2},
        "ag_news": {8: 82.9, 64: 88.0},
    }

    for dataset, exp in results["experiments"].items():
        print(f"Dataset: {dataset.upper()}")
        print(f"  {exp['description']}")
        for n_str, metrics in exp["results_by_n"].items():
            n = int(n_str)
            if "error" in metrics:
                print(f"  N={n:3d}: ERROR - {metrics['error'][:60]}")
            else:
                acc = metrics["accuracy"] * 100
                paper_ref = paper_results.get(dataset, {}).get(n, None)
                ref_str = f" (paper: {paper_ref:.1f}%)" if paper_ref else ""
                print(f"  N={n:3d}: {acc:.2f}%{ref_str}")
        print()


if __name__ == "__main__":
    main()

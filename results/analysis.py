"""
SetFit Results Analysis and Visualization
Generates comparison plots of SetFit vs baseline performance.
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_results(path="setfit_results.json"):
    with open(path) as f:
        return json.load(f)


def plot_results(results, save_path="setfit_comparison.png"):
    experiments = results["experiments"]
    datasets = list(experiments.keys())
    n_values = [8, 64]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "SetFit: Efficient Few-Shot Learning Without Prompts\n"
        "arXiv:2209.11055 — Tunstall et al. (2022)",
        fontsize=13, fontweight="bold", y=1.01
    )

    colors = {"SetFit": "#2196F3", "Baseline": "#FF7043", "Paper (MPNet)": "#4CAF50"}
    x = np.arange(len(n_values))
    width = 0.25

    for ax, dataset in zip(axes, datasets):
        setfit_accs, baseline_accs, paper_accs = [], [], []

        for n in n_values:
            m = experiments[dataset].get(str(n), {})
            setfit_accs.append(m.get("setfit_accuracy", 0) * 100)
            baseline_accs.append(m.get("baseline_accuracy", 0) * 100)
            paper = m.get("paper_reference_mpnet")
            paper_accs.append(paper if paper else 0)

        b1 = ax.bar(x - width, baseline_accs, width, label="FineTune (Baseline)",
                    color=colors["Baseline"], alpha=0.8, edgecolor="white")
        b2 = ax.bar(x, setfit_accs, width, label="SetFit (Ours - TF-IDF)",
                    color=colors["SetFit"], alpha=0.8, edgecolor="white")
        b3 = ax.bar(x + width, paper_accs, width, label="SetFit_MPNet (Paper)",
                    color=colors["Paper (MPNet)"], alpha=0.8, edgecolor="white")

        # Value labels
        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

        ax.set_xlabel("Number of training samples per class (N)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(f"{dataset} Classification", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in n_values])
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Note
    fig.text(0.5, -0.02,
             "Note: 'Ours' uses TF-IDF+SVD encoder (offline). "
             "Paper uses paraphrase-mpnet-base-v2 (110M param pretrained ST).",
             ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")


def print_comparison_table(results):
    print("\n" + "=" * 75)
    print("DETAILED COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Dataset':<12} {'N':>4}  {'SetFit':>8}  {'Baseline':>9}  "
          f"{'Δ':>7}  {'Paper MPNet':>12}  {'Gap to Paper':>12}")
    print("-" * 75)

    for dataset, exp in results["experiments"].items():
        for n_str, m in exp.items():
            if "error" in m:
                continue
            n = int(n_str)
            sf = m["setfit_accuracy"] * 100
            bl = m["baseline_accuracy"] * 100
            delta = m["improvement"]
            paper = m.get("paper_reference_mpnet")
            gap = sf - paper if paper else None

            paper_str = f"{paper:.1f}%" if paper else "N/A"
            gap_str = f"{gap:+.1f}%" if gap is not None else "N/A"
            print(f"{dataset:<12} {n:>4}  {sf:>7.2f}%  {bl:>8.2f}%  "
                  f"{delta:>+7.2f}%  {paper_str:>12}  {gap_str:>12}")
    print("=" * 75)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    results = load_results()
    print_comparison_table(results)
    try:
        plot_results(results)
    except Exception as e:
        print(f"Plot error (matplotlib may not be available): {e}")

"""
evaluation/plot_results.py
──────────────────────────
Reads outputs/results.json (produced by evaluate_robustness.py) and generates:

  1. Bar chart   – Clean accuracy comparison (Teacher / Baseline / KD)
  2. Line chart  – Accuracy vs severity for each corruption
  3. Bar chart   – Robustness gap comparison (Baseline vs KD)
  4. Heatmap     – Full accuracy grid (model × condition)

All figures saved under outputs/figures/
"""

import json
import os

import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_PATH = "outputs/results.json"
FIGURES_DIR  = "outputs/figures"
SEVERITIES   = [1, 3, 5]
CORRUPTIONS  = ["gaussian_noise", "blur", "brightness", "contrast"]
CORR_LABELS  = {"gaussian_noise": "Gaussian Noise",
                "blur":           "Blur",
                "brightness":     "Brightness",
                "contrast":       "Contrast"}

# colour palette
COLORS = {
    "Teacher (ResNet-34)":    "#4C72B0",
    "Baseline (ResNet-18)":   "#DD8452",
    "KD Student (ResNet-18)": "#55A868",
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def savefig(name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 1. Clean accuracy bar chart ───────────────────────────────────────────────

def plot_clean_accuracy(results: dict):
    models = list(results.keys())
    accs   = [results[m]["clean"] for m in models]
    colors = [COLORS.get(m, "#999") for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, accs, color=colors, width=0.5, zorder=3)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)

    ax.set_title("Clean Test Accuracy — CIFAR-100", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(accs) * 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    plt.tight_layout()
    savefig("1_clean_accuracy.png")


# ── 2. Accuracy vs severity line charts ──────────────────────────────────────

def plot_severity_curves(results: dict):
    fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(18, 4), sharey=False)

    for ax, corruption in zip(axes, CORRUPTIONS):
        for model_name, data in results.items():
            if corruption not in data:
                continue
            accs = [data[corruption].get(str(s), data[corruption].get(s, None))
                    for s in SEVERITIES]
            if None in accs:
                continue
            ax.plot(SEVERITIES, accs,
                    marker="o",
                    label=model_name,
                    color=COLORS.get(model_name, "#999"),
                    linewidth=2)

        ax.set_title(CORR_LABELS[corruption], fontsize=11, fontweight="bold")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(SEVERITIES)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)

    fig.suptitle("Accuracy vs Corruption Severity", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("2_severity_curves.png")


# ── 3. Robustness gap bar chart ───────────────────────────────────────────────

def plot_robustness_gap(results: dict):
    # Only compare Baseline and KD student
    models_of_interest = ["Baseline (ResNet-18)", "KD Student (ResNet-18)"]
    models  = [m for m in models_of_interest if m in results]
    gaps    = [results[m]["robustness_gap"] for m in models]
    colors  = [COLORS.get(m, "#999") for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, gaps, color=colors, width=0.4, zorder=3)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11)

    ax.set_title("Robustness Gap\n(Clean Acc − Mean Corrupted Acc)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gap (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    # Lower gap = more robust
    ax.annotate("← more robust", xy=(0.98, 0.05), xycoords="axes fraction",
                ha="right", fontsize=8, color="gray")
    plt.tight_layout()
    savefig("3_robustness_gap.png")


# ── 4. Heatmap ───────────────────────────────────────────────────────────────

def plot_heatmap(results: dict):
    models = list(results.keys())

    # build column labels
    col_labels = ["Clean"]
    for c in CORRUPTIONS:
        for s in SEVERITIES:
            col_labels.append(f"{CORR_LABELS[c]}\nsev={s}")

    # build matrix
    matrix = []
    for m in models:
        row = [results[m].get("clean", 0)]
        for c in CORRUPTIONS:
            for s in SEVERITIES:
                val = results[m].get(c, {}).get(str(s),
                      results[m].get(c, {}).get(s, 0))
                row.append(val)
        matrix.append(row)

    data = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(14, len(models) * 1.4 + 1.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(len(models)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{data[i, j]:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="black" if 20 < data[i, j] < 80 else "white")

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_title("Accuracy Heatmap — All Models × All Conditions",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    savefig("4_heatmap.png")


# ── Print summary table ───────────────────────────────────────────────────────

def print_summary_table(results: dict):
    print("\n" + "=" * 55)
    print("SUMMARY TABLE")
    print("=" * 55)
    print(f"{'Model':<28} {'Clean':>6} {'Gap':>6}")
    print("-" * 55)
    for m, data in results.items():
        print(f"{m:<28} {data.get('clean', 0):>5.2f}%  {data.get('robustness_gap', 0):>5.2f}%")
    print("=" * 55)
    print("Gap = Clean Acc − Mean Corrupted Acc (lower = more robust)\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"[ERROR] {RESULTS_PATH} not found.")
        print("Run  python -m evaluation.evaluate_robustness  first.")
        return

    results = load_results(RESULTS_PATH)
    print_summary_table(results)

    print("Generating figures …")
    plot_clean_accuracy(results)
    plot_severity_curves(results)
    plot_robustness_gap(results)
    plot_heatmap(results)
    print("Done! All figures saved to outputs/figures/")


if __name__ == "__main__":
    main()

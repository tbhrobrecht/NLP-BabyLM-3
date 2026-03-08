"""
Generate two line plot graphs comparing BLiMP evaluation results:
  1. All files in initial_parsed_blimp_results/
  2. parsed_blimp_results/norm_pinyin_abbreviations_blimp_results.txt
     vs. parsed_blimp_results/initials_blimp_results.csv
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(filepath: str) -> dict[str, float]:
    """Load a phenomenon->accuracy mapping from a CSV/TXT results file."""
    results = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["phenomenon"].strip()] = float(row["accuracy"])
    return results


def make_line_plot(
    datasets: dict[str, dict[str, float]],
    title: str,
    output_path: str,
) -> None:
    """
    Plot a line graph where:
      - x-axis: phenomenon names (shared across all datasets)
      - y-axis: accuracy (0–1)
      - one line per dataset (label = filename stem)
    """
    # Collect phenomena that appear in ALL datasets, preserve insertion order
    # of the first file while filtering to common keys.
    all_keys = [list(d.keys()) for d in datasets.values()]
    # Start from the union, then keep only those present in every dataset
    common_phenomena = [
        p for p in all_keys[0] if all(p in d for d in datasets.values())
    ]

    fig, ax = plt.subplots(figsize=(max(14, len(common_phenomena) * 0.35), 6))

    for label, data in datasets.items():
        y = [data[p] for p in common_phenomena]
        ax.plot(range(len(common_phenomena)), y, marker="o", markersize=4, label=label, linewidth=1.5)

    ax.set_xticks(range(len(common_phenomena)))
    ax.set_xticklabels(common_phenomena, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Phenomenon")
    # ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def make_scatter_plot(
    datasets: dict[str, dict[str, float]],
    title: str,
    output_path: str,
) -> None:
    """
    Scatter plot where:
      - x-axis: phenomenon names (shared across all datasets)
      - y-axis: accuracy (0–1)
      - one series of markers per dataset (label = filename stem)
    """
    all_keys = [list(d.keys()) for d in datasets.values()]
    common_phenomena = [
        p for p in all_keys[0] if all(p in d for d in datasets.values())
    ]

    fig, ax = plt.subplots(figsize=(max(14, len(common_phenomena) * 0.35), 6))

    for label, data in datasets.items():
        y = [data[p] for p in common_phenomena]
        ax.scatter(range(len(common_phenomena)), y, s=40, label=label, zorder=3)

    ax.set_xticks(range(len(common_phenomena)))
    ax.set_xticklabels(common_phenomena, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Phenomenon")
    # ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graph 1 – all files in initial_parsed_blimp_results/
# ---------------------------------------------------------------------------

INITIAL_DIR = os.path.join(os.path.dirname(__file__), "initial_parsed_blimp_results")

INITIAL_LABEL_MAP = {
    "initials_2xpinyin_blimp": "A2xPinyin",
    "initials_blimp_results": "Abbreviations",
    "initials_hanzi_blimp": "AHanzi",
    "initials_pinyin_blimp": "APinyin",
}

initial_datasets: dict[str, dict[str, float]] = {}
for fname in sorted(os.listdir(INITIAL_DIR)):
    if fname.endswith((".csv", ".txt")):
        path = os.path.join(INITIAL_DIR, fname)
        stem = os.path.splitext(fname)[0]
        label = INITIAL_LABEL_MAP.get(stem, stem)
        initial_datasets[label] = load_results(path)

make_line_plot(
    datasets=initial_datasets,
    title="BLiMP Accuracy – initial_parsed_blimp_results",
    output_path=os.path.join(os.path.dirname(__file__), "plot_initial_comparison.png"),
)

# ---------------------------------------------------------------------------
# Graph 2 – norm_pinyin_abbreviations vs initials (parsed_blimp_results/)
# ---------------------------------------------------------------------------

PARSED_DIR = os.path.join(os.path.dirname(__file__), "parsed_blimp_results")

comparison_datasets = {
    "Control": load_results(
        os.path.join(PARSED_DIR, "norm_pinyin_abbreviations_blimp_results.txt")
    ),
    "Abbreviations": load_results(
        os.path.join(PARSED_DIR, "initials_blimp_results.csv")
    ),
}

make_line_plot(
    datasets=comparison_datasets,
    title="BLiMP Accuracy – norm_pinyin_abbreviations vs initials",
    output_path=os.path.join(os.path.dirname(__file__), "plot_norm_pinyin_vs_initials.png"),
)

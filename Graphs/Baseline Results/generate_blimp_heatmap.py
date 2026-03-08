#!/usr/bin/env python3
"""
generate_blimp_heatmap.py
-------------------------
Generates a paper-grade A4 heatmap of BLiMP accuracy results from
the parsed_results_all directory.

Usage:
    python generate_blimp_heatmap.py
    python generate_blimp_heatmap.py --results_dir parsed_results_all --output blimp_heatmap.pdf
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────

# Ordered: (filename, display_label)
# Grouped: initials-based models | standalone character/pinyin models
MODELS = [
    ("initials_blimp_results.csv",                "Abbreviations"),
    ("initials_hanzi_blimp.txt",                  "AHanzi"),
    ("initials_pinyin_blimp.txt",                 "APinyin"),
    ("initials_2xpinyin_blimp.txt",               "A2×Pinyin"),
    ("norm_hanzi_blimp_results.txt",              "Hanzi\n(Norm)"),
    ("trunc_hanzi_blimp_results.txt",             "Hanzi\n(Trunc)"),
    ("norm_pinyin_blimp_results.txt",             "Pinyin\n(Norm)"),
    ("trunc_pinyin_blimp_results.txt",            "Pinyin\n(Trunc)"),
    ("norm_pinyin_abbreviations_blimp_results.txt", "Control"),
]

# A4 portrait in inches
A4_W, A4_H = 8.27, 11.69


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(results_dir: Path) -> pd.DataFrame:
    dfs = {}
    for fname, label in MODELS:
        p = results_dir / fname
        if not p.exists():
            print(f"[WARN] File not found, skipping: {p}")
            continue
        series = pd.read_csv(p).set_index("phenomenon")["accuracy"]
        dfs[label] = series

    col_order = [label for _, label in MODELS if label in dfs]
    combined = pd.DataFrame(dfs, columns=col_order)
    combined = combined.sort_index()
    combined.index = [idx.replace("_", " ") for idx in combined.index]
    return combined


def plot_heatmap(data: pd.DataFrame, output: Path) -> None:
    n_rows, n_cols = data.shape

    # Scale y-axis font to available space (min 4pt, max 5.5pt)
    row_fs = max(4.0, min(5.5, 570 / n_rows))

    # Use vector-font-embedding settings for print quality
    matplotlib.rcParams.update({
        "font.family":  "serif",
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    fig, ax = plt.subplots(figsize=(A4_W, A4_H))

    # ── Heatmap ───────────────────────────────────────────────────────────────
    sns.heatmap(
        data,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        square=False,
        cbar_kws={
            "shrink":  0.30,
            "pad":     0.03,
            "aspect":  20,
            # "label":   "Accuracy",
            "ticks":   [0.0, 0.25, 0.50, 0.75, 1.0],
        },
        annot=False,
    )

    # ── X-axis (models) at top ────────────────────────────────────────────────
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(
        data.columns,
        rotation=0,
        ha="center",
        fontsize=6.5,
        multialignment="center",
        linespacing=1.3,
    )
    ax.tick_params(axis="x", which="both", length=0, pad=5)

    # ── Y-axis (phenomena) ───────────────────────────────────────────────────
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(
        data.index,
        rotation=0,
        ha="right",
        fontsize=row_fs,
        va="center",
    )
    ax.tick_params(axis="y", which="both", length=0, pad=3)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # ── Subtle horizontal dividers every 10 rows ─────────────────────────────
    for i in range(10, n_rows, 10):
        ax.axhline(i, color="#aaaaaa", linewidth=0.4, linestyle=":")

    # ── Vertical divider between initials group and standalone models ─────────
    initials_cols = [lbl for _, lbl in MODELS
                     if lbl.startswith("A") and lbl in data.columns]
    n_init = len(initials_cols)
    if 0 < n_init < n_cols:
        ax.axvline(n_init, color="#555555", linewidth=0.8, linestyle="--")

    # ── Colorbar styling ──────────────────────────────────────────────────────
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6.5)
    # cbar.set_label("Accuracy", fontsize=7)
    # Draw a tick mark at chance level (0.5)
    cbar.ax.axhline(0.5, color="#333333", linewidth=0.8, linestyle="--")
    # cbar.ax.text(
    #     1.8, 0.5, "chance", va="center", ha="left",
    #     fontsize=5.5, color="#333333",
    #     transform=cbar.ax.get_yaxis_transform(),
    # )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "ZhoBLiMP Accuracy Heatmap",
        fontsize=9,
        fontweight="bold",
        pad=14,
    )

    # ── Layout & save ─────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.5)
    suffix = output.suffix.lower().lstrip(".")
    fmt = suffix if suffix in {"pdf", "png", "svg", "eps"} else "pdf"
    fig.savefig(output, dpi=300, bbox_inches="tight", format=fmt)
    print(f"Saved → {output}  ({n_rows} phenomena × {n_cols} models)")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate BLiMP accuracy heatmap.")
    parser.add_argument("--results_dir", default="parsed_results_all",
                        help="Directory containing result .txt/.csv files.")
    parser.add_argument("--output", default="blimp_heatmap.pdf",
                        help="Output file (supports .pdf, .png, .svg).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output = Path(args.output)

    data = load_data(results_dir)
    if data.empty:
        print("[ERROR] No data loaded. Check that results_dir is correct.")
        return

    plot_heatmap(data, output)


if __name__ == "__main__":
    main()

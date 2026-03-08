"""
make_blimp_table.py
-------------------
Reads BLiMP result files and outputs a MuZero-style LaTeX table for Overleaf.

Usage:
    python make_blimp_table.py [--results_dir PATH] [--output_dir PATH] [--ties_all_bold]

To include the generated table in Overleaf:
    \\usepackage{booktabs}
    \\usepackage{multirow}
    ...
    \\input{tables/blimp_muzero_style.tex}
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MODEL_DISPLAY_NAMES: dict[str, str] = {
#     "initials":           "Initials",
#     "pinyin_abbrev_norm": "Pinyin-Abbrev",
#     "hanzi_norm":         "Hanzi (norm)",
#     "hanzi_trunc":        "Hanzi (trunc)",
#     "pinyin_norm":        "Pinyin (norm)",
#     "pinyin_trunc":       "Pinyin (trunc)",
# }
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "initials":         "Initials",
    "hanzi":            "Hanzi",
    "pinyin":           "Pinyin",
    "2xpinyin":         "2xPinyin",
}

# Explicit order for model columns
# MODEL_COLS: list[str] = [
#     "initials",
#     "pinyin_abbrev_norm",
#     "hanzi_norm",
#     "hanzi_trunc",
#     "pinyin_norm",
#     "pinyin_trunc",
# ]
MODEL_COLS: list[str] = [
    "initials",
    "hanzi",
    "pinyin",
    "2xpinyin"
]
SUMMARY_COLS: list[str] = ["row_mean", "row_median"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_result_file(path: Path, col_name: str) -> pd.DataFrame:
    """Load a comma-separated result file (with header phenomenon,accuracy).

    Returns a DataFrame with columns [phenomenon, <col_name>].
    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}\n"
            f"Expected a CSV-like file with header 'phenomenon,accuracy'."
        )
    df = pd.read_csv(path, dtype={"phenomenon": str, "accuracy": float})
    if "phenomenon" not in df.columns or "accuracy" not in df.columns:
        raise ValueError(
            f"File {path} must have columns 'phenomenon' and 'accuracy', "
            f"but found: {list(df.columns)}"
        )
    return df.rename(columns={"accuracy": col_name})[["phenomenon", col_name]]


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all six result files and merge them on 'phenomenon'."""
    # file_map: dict[str, str] = {
    #     "initials_blimp_results.csv":             "initials",
    #     "norm_hanzi_blimp_results.txt":           "hanzi_norm",
    #     "trunc_hanzi_blimp_results.txt":          "hanzi_trunc",
    #     "norm_pinyin_blimp_results.txt":          "pinyin_norm",
    #     "trunc_pinyin_blimp_results.txt":         "pinyin_trunc",
    #     "norm_pinyin_abbreviations_blimp_results.txt": "pinyin_abbrev_norm",
    # }
    file_map: dict[str, str] = {
        "initials_blimp_results.csv":         "initials",
        "initials_hanzi_blimp.txt":           "hanzi",
        "initials_pinyin_blimp.txt":          "pinyin",
        "initials_2xpinyin_blimp.txt":        "2xpinyin",
    }

    merged: Optional[pd.DataFrame] = None
    for filename, col_name in file_map.items():
        df = load_result_file(results_dir / filename, col_name)
        merged = df if merged is None else merged.merge(df, on="phenomenon", how="outer")

    assert merged is not None
    merged = merged.sort_values("phenomenon").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def add_row_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Append row_mean and row_median columns (ignoring NaN)."""
    num = df[MODEL_COLS].astype(float)
    df["row_mean"]   = num.mean(axis=1)
    df["row_median"] = num.median(axis=1)
    return df


def build_summary_rows(df: pd.DataFrame, ties_all_bold: bool = True) -> pd.DataFrame:
    """Return a small DataFrame with the Mean, Median and #Best summary rows."""
    model_vals = df[MODEL_COLS].astype(float)

    mean_row   = model_vals.mean().rename("Mean")
    median_row = model_vals.median().rename("Median")

    # #Best: per phenomenon, count how many times a model is tied for max
    row_max = model_vals.max(axis=1)
    best_counts = (model_vals.eq(row_max, axis=0)).sum()
    best_row = best_counts.rename("#Best")

    # Build summary DataFrame columns in order
    summary = pd.DataFrame([mean_row, median_row, best_row])
    summary.index.name = "phenomenon"
    summary = summary.reset_index()

    # Fill in summary cols with NaN so we can concat cleanly
    summary["row_mean"]   = float("nan")
    summary["row_median"] = float("nan")

    return summary[["phenomenon"] + MODEL_COLS + SUMMARY_COLS]


def _count_threshold(
    model_vals: pd.DataFrame,
    col: str,
    high_thresh: float,
    low_thresh: float,
) -> tuple[int, int]:
    """Return (count >= high_thresh, count <= low_thresh) for one model column."""
    vals = model_vals[col].dropna()
    return int((vals >= high_thresh).sum()), int((vals <= low_thresh).sum())


def build_count_rows(df: pd.DataFrame) -> list[dict]:
    """Build individual threshold-count rows for summary: 0%, <=5%, >=95%, 100%."""
    model_vals = df[MODEL_COLS].astype(float)

    def count_row(label: str, fn) -> dict:
        row = {"phenomenon": label}
        for col in MODEL_COLS:
            row[col] = str(fn(model_vals[col].dropna()))
        return row

    rows = [
        count_row(r"0\% accuracy", lambda s: (s == 0.0).sum()),
        count_row(r"${\leq}5\%$ accuracy", lambda s: (s <= 0.05).sum()),
        count_row(r"${\geq}95\%$ accuracy", lambda s: (s >= 0.95).sum()),
        count_row(r"100\% accuracy", lambda s: (s == 1.0).sum()),
    ]
    return rows


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def escape_tex(text: str) -> str:
    """Escape LaTeX special characters in plain text (not in commands)."""
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def monospace(text: str) -> str:
    """Wrap a phenomenon name in \\texttt{} with escaped underscores."""
    return r"\texttt{" + escape_tex(text) + "}"


def fmt_val(val: float, is_best_row: bool = False) -> str:
    """Format a float to 2 decimal places, or '—' for NaN."""
    if math.isnan(val):
        return "---"
    formatted = f"{val:.2f}"
    if is_best_row:
        return r"\textbf{" + formatted + "}"
    return formatted


def fmt_best_count(val: float) -> str:
    """Format an integer #Best count (stored as float after concat)."""
    if math.isnan(val):
        return "---"
    return str(int(val))


def _render_count_row(label: str, row: dict) -> str:
    """Render a threshold-count row (values are 'X / Y' strings)."""
    cells = [r"\textbf{" + label + "}"]
    for col in MODEL_COLS:
        cells.append(str(row.get(col, "---")))
    cells += ["---", "---"]  # no row_mean / row_median
    return " & ".join(cells) + r" \\" 


def _render_summary_row(label: str, row: "pd.Series") -> str:
    """Render a Mean/Median/#Best summary row, bolding the best model value(s)."""
    cells_model: list[str] = []
    if label == "#Best":
        raw_vals = [
            float(row[c]) if not pd.isna(row[c]) else float("nan")
            for c in MODEL_COLS
        ]
        max_count = max((v for v in raw_vals if not math.isnan(v)), default=float("nan"))
        for val in raw_vals:
            if math.isnan(val):
                cells_model.append("---")
            else:
                count_str = str(int(val))
                cells_model.append(
                    r"\textbf{" + count_str + "}" if math.isclose(val, max_count) else count_str
                )
    else:
        raw_vals = [
            float(row[c]) if not pd.isna(row[c]) else float("nan")
            for c in MODEL_COLS
        ]
        max_val = max((v for v in raw_vals if not math.isnan(v)), default=float("nan"))
        for val in raw_vals:
            is_best = (not math.isnan(val)) and math.isclose(val, max_val, rel_tol=1e-9)
            cells_model.append(fmt_val(val, is_best_row=is_best))
    cells_model += ["---", "---"]  # no row_mean / row_median for summary rows
    return r"\textbf{" + escape_tex(label) + "} & " + " & ".join(cells_model) + r" \\"


def build_latex_table(
    data_rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
    count_rows: list[dict],
    ties_all_bold: bool = True,
) -> str:
    """Produce a complete LaTeX longtable environment as a string."""
    n_model_cols   = len(MODEL_COLS)
    n_summary_cols = len(SUMMARY_COLS)
    total_cols     = 1 + n_model_cols + n_summary_cols

    col_spec       = "l" + "r" * n_model_cols + "rr"
    header_display = [MODEL_DISPLAY_NAMES[c] for c in MODEL_COLS]
    summary_display = ["Mean", "Median"]

    model_start  = 2
    model_end    = 1 + n_model_cols
    summary_start = model_end + 1
    summary_end   = total_cols

    def header_rows() -> list[str]:
        model_span   = r"\multicolumn{" + str(n_model_cols)   + r"}{c}{Models}"
        summary_span = r"\multicolumn{" + str(n_summary_cols) + r"}{c}{Summary}"
        return [
            r"\toprule",
            r"\textbf{Phenomenon} & " + model_span + " & " + summary_span + r" \\",
            rf"\cmidrule(lr){{{model_start}-{model_end}}}\cmidrule(lr){{{summary_start}-{summary_end}}}",
            r"\textbf{Phenomenon} & " + " & ".join(
                [r"\textbf{" + h + "}" for h in header_display + summary_display]
            ) + r" \\",
            r"\midrule",
        ]

    lines: list[str] = []

    # ---- preamble comments ----
    lines += [
        "% Auto-generated by make_blimp_table.py",
        "% Include in Overleaf with:",
        "%   \\usepackage{booktabs}",
        "%   \\usepackage{longtable}",
        "%   \\input{tables/blimp_muzero_style.tex}",
        "",
    ]

    lines.append(r"\begin{longtable}{" + col_spec + r"}")

    # caption + label before \endfirsthead
    lines.append(
        r"\caption{BLiMP accuracy per phenomenon for each model representation. "
        r"Best result(s) per row are \textbf{bolded}. "
        r"Row Mean and Median are computed across all six models.}"
        r"\label{tab:blimp_results} \\"
    )

    # First-page header
    lines += header_rows()
    lines.append(r"\endfirsthead")

    # Continuation header
    lines.append(
        r"\multicolumn{" + str(total_cols) + r"}{c}{\tablename\ \thetable{} -- continued} \\"
    )
    lines += header_rows()
    lines.append(r"\endhead")

    # Footer on non-last pages
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{" + str(total_cols) + r"}{r}{\textit{Continued on next page}} \\"
    )
    lines.append(r"\endfoot")

    # Last-page footer
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    # ---- Data rows ----
    model_data   = data_rows[MODEL_COLS].astype(float)
    row_max_vals = model_data.max(axis=1)

    for _, row in data_rows.iterrows():
        row_max = row_max_vals[row.name]
        cells: list[str] = [monospace(str(row["phenomenon"]))]
        for col in MODEL_COLS:
            val = float(row[col]) if not pd.isna(row[col]) else float("nan")
            is_best = (not math.isnan(val)) and math.isclose(val, row_max, rel_tol=1e-9)
            cells.append(fmt_val(val, is_best_row=is_best))
        for col in SUMMARY_COLS:
            val = float(row[col]) if not pd.isna(row[col]) else float("nan")
            cells.append(fmt_val(val))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # ---- Bottom summary rows: count rows first, then Mean/Median/#Best ----
    for crow in count_rows:
        lines.append(_render_count_row(crow["phenomenon"], crow))

    for _, row in summary_rows.iterrows():
        lines.append(_render_summary_row(str(row["phenomenon"]), row))

    lines.append(r"\end{longtable}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a MuZero-style LaTeX BLiMP results table."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("initial_parsed_blimp_results"),
        help="Directory containing the six result files (default: initial_parsed_blimp_results/).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tables"),
        help="Directory for output files (default: tables/).",
    )
    parser.add_argument(
        "--ties_all_bold",
        action="store_true",
        default=True,
        help="If set, all tied-best models are bolded per row (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and merge
    print(f"Loading result files from: {args.results_dir}")
    merged = load_all_results(args.results_dir)
    print(f"  Loaded {len(merged)} phenomena across {len(MODEL_COLS)} models.")

    # 2. Row summaries
    merged = add_row_summaries(merged)

    # 3. Debug CSV
    csv_path = args.output_dir / "blimp_merged.csv"
    merged.to_csv(csv_path, index=False)
    print(f"  Merged CSV written to: {csv_path}")

    # 4. Summary rows
    summary = build_summary_rows(merged, ties_all_bold=args.ties_all_bold)
    counts  = build_count_rows(merged)

    # 5. Build LaTeX
    latex = build_latex_table(merged, summary, counts, ties_all_bold=args.ties_all_bold)

    # 6. Write .tex
    tex_path = args.output_dir / "blimp_muzero_style.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"  LaTeX table written to: {tex_path}")
    print(
        "\nTo include in Overleaf:\n"
        "  \\usepackage{booktabs}\n"
        "  \\usepackage{longtable}\n"
        "  ...\n"
        "  \\input{tables/blimp_muzero_style.tex}"
    )


if __name__ == "__main__":
    main()

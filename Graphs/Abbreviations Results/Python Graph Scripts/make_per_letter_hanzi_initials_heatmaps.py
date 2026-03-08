#!/usr/bin/env python3
"""
Generate one log-scale heatmap per pinyin initial letter (a-z).

For each letter L, plots:
- Y-axis: Hanzi terms whose pinyin-initials pattern starts with L
- X-axis: pinyin initial patterns observed for those terms
- Cell: co-occurrence frequency
- Log-scale colormap like the example

Example: "武汉大学" -> pinyin initials "whdx" -> bucket letter 'w'
"""

import argparse
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm
import jieba
from pypinyin import pinyin, Style


# Regex to detect CJK characters (Hanzi)
CJK_PATTERN = re.compile(r'[\u4e00-\u9fff]')


def set_cjk_font():
    """
    Set matplotlib font to support CJK (Chinese) characters on Windows.
    Tries common fonts in order and sets the first available one.
    """
    fonts_to_try = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC"
    ]
    
    for font_name in fonts_to_try:
        try:
            font_path = fm.findfont(font_name, fallback_to_default=False)
            # Check if actually found (not just returned default)
            if font_path and font_name.lower() in font_path.lower():
                plt.rcParams["font.family"] = font_name
                plt.rcParams["axes.unicode_minus"] = False
                print(f"Set CJK font: {font_name}")
                return
        except Exception:
            continue
    
    print("WARNING: No CJK font found. Chinese characters may not display correctly.")
    print("Consider installing: Microsoft YaHei, SimHei, or Noto Sans CJK SC")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-letter Hanzi term heatmaps conditioned on pinyin initial patterns"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="corpus.csv",
        help="Path to corpus CSV file with 'text' column (default: corpus.csv)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/per_letter_heatmaps",
        help="Output directory for heatmap images (default: outputs/per_letter_heatmaps)"
    )
    parser.add_argument(
        "--top_terms",
        type=int,
        default=30,
        help="Number of top terms to include per letter (default: 30)"
    )
    parser.add_argument(
        "--top_patterns",
        type=int,
        default=20,
        help="Number of top patterns to include per letter (default: 20)"
    )
    parser.add_argument(
        "--min_term_total",
        type=int,
        default=20,
        help="Minimum total frequency for a term to be included (default: 20)"
    )
    parser.add_argument(
        "--min_pattern_total",
        type=int,
        default=20,
        help="Minimum total frequency for a pattern to be included (default: 20)"
    )
    parser.add_argument(
        "--cell_threshold_ratio",
        type=float,
        default=0.05,
        help="Mask cells with freq < ratio * max_cell (default: 0.05)"
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        choices=["global", "per_column", "none"],
        default="global",
        help="Thresholding mode: 'global' (ratio of global max), 'per_column' (ratio per column max), 'none' (no threshold, only mask zeros) (default: global)"
    )
    parser.add_argument(
        "--letters",
        type=str,
        default=None,
        help="Comma-separated list of letters to plot (e.g., 'e,w,n'), default: all"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (only for small number of letters)"
    )
    parser.add_argument(
        "--max_letters_to_show",
        type=int,
        default=3,
        help="Maximum number of letters to show interactively (default: 3)"
    )
    return parser.parse_args()


def is_cjk(text):
    """Check if text contains at least one CJK (Hanzi) character."""
    return bool(CJK_PATTERN.search(text))


def term_to_initials(term):
    """
    Convert a Hanzi term to its pinyin initials pattern.
    
    Example: "武汉大学" -> "whdx"
    
    Args:
        term: Chinese text string
    
    Returns:
        initials pattern string (lowercase), or None if conversion fails
    """
    try:
        # Get pinyin syllables with NORMAL style (tone marks removed)
        syllables_list = pinyin(term, style=Style.NORMAL, strict=False, errors='ignore')
        
        # Extract first letter of each syllable
        initials = []
        for syl_list in syllables_list:
            if syl_list and syl_list[0]:
                first_char = syl_list[0][0].lower()
                if first_char.isalpha():
                    initials.append(first_char)
        
        if initials:
            return ''.join(initials)
        else:
            return None
    except Exception:
        return None


def read_and_process_corpus(csv_path):
    """
    Read corpus CSV and extract Hanzi terms grouped by pinyin initial letter.
    
    Returns:
        cooc_by_letter: dict[str, Counter] mapping letter -> {(term, pattern): count}
        term_total_by_letter: dict[str, Counter] mapping letter -> {term: count}
        pattern_total_by_letter: dict[str, Counter] mapping letter -> {pattern: count}
        stats: dict with processing statistics
    """
    print(f"Reading corpus from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        sys.exit(1)
    
    rows_read = len(df)
    print(f"Read {rows_read:,} rows")
    
    # Filter to Chinese text if possible
    rows_kept = rows_read
    if 'language' in df.columns and 'script' in df.columns:
        df = df[(df['language'] == 'zho') & (df['script'] == 'Hans')]
        rows_kept = len(df)
        print(f"Filtered to {rows_kept:,} Chinese (zho/Hans) rows")
    elif 'language' in df.columns:
        df = df[df['language'] == 'zho']
        rows_kept = len(df)
        print(f"Filtered to {rows_kept:,} Chinese (zho) rows")
    else:
        print("No language/script columns found, processing all rows")
    
    if rows_kept == 0:
        print("ERROR: No rows remaining after filtering")
        sys.exit(1)
    
    # Check for text column
    if 'text' not in df.columns:
        print(f"ERROR: 'text' column not found in CSV. Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Process text to extract terms grouped by bucket letter
    cooc_by_letter = defaultdict(Counter)
    term_total_by_letter = defaultdict(Counter)
    pattern_total_by_letter = defaultdict(Counter)
    
    total_jieba_terms = 0
    rows_processed = 0
    
    print("\nProcessing text with jieba segmentation...")
    
    for idx, row in df.iterrows():
        text = row['text']
        if pd.isna(text) or not isinstance(text, str):
            continue
        
        # Segment with jieba
        terms = jieba.lcut(text)
        
        for term in terms:
            # Skip empty or whitespace-only terms
            term = term.strip()
            if not term:
                continue
            
            # Only keep terms with at least one CJK character
            if not is_cjk(term):
                continue
            
            total_jieba_terms += 1
            
            # Get pinyin initials pattern
            initials_pattern = term_to_initials(term)
            if initials_pattern:
                # Bucket letter is the FIRST letter of the initials pattern
                bucket_letter = initials_pattern[0]
                
                # Update counts
                cooc_by_letter[bucket_letter][(term, initials_pattern)] += 1
                term_total_by_letter[bucket_letter][term] += 1
                pattern_total_by_letter[bucket_letter][initials_pattern] += 1
        
        rows_processed += 1
        if rows_processed % 10000 == 0:
            print(f"  Processed {rows_processed:,} rows, {total_jieba_terms:,} CJK terms...")
    
    print(f"Finished processing {rows_processed:,} rows")
    
    stats = {
        'rows_read': rows_read,
        'rows_kept': rows_kept,
        'total_jieba_terms': total_jieba_terms
    }
    
    return cooc_by_letter, term_total_by_letter, pattern_total_by_letter, stats


def filter_and_select_for_letter(letter, cooc_counter, term_counter, pattern_counter, args):
    """
    Filter and select terms and patterns for a single letter.
    
    FIX: Select patterns conditioned on selected_terms to avoid empty columns.
    
    Returns:
        selected_terms: list of term strings
        selected_patterns: list of pattern strings
        or (None, None) if insufficient data
    """
    # Step 1: Filter and select terms (as before)
    terms_above_min = [t for t, cnt in term_counter.items() 
                       if cnt >= args.min_term_total]
    
    if len(terms_above_min) < 2:
        print(f"  Letter {letter}: Insufficient terms after filtering "
              f"({len(terms_above_min)} terms) - skipping")
        return None, None
    
    # Select top N terms by frequency
    selected_terms = [t for t, _ in term_counter.most_common() 
                      if t in terms_above_min][:args.top_terms]
    
    if len(selected_terms) < 2:
        print(f"  Letter {letter}: Insufficient terms after top-N selection - skipping")
        return None, None
    
    # Step 2: Recompute pattern totals restricted to selected_terms
    selected_terms_set = set(selected_terms)
    restricted_pattern_counter = Counter()
    
    for (term, pattern), cnt in cooc_counter.items():
        if term in selected_terms_set:
            restricted_pattern_counter[pattern] += cnt
    
    # Step 3: Filter patterns by minimum total (on restricted counter)
    patterns_above_min = [p for p, cnt in restricted_pattern_counter.items() 
                          if cnt >= args.min_pattern_total]
    
    if len(patterns_above_min) < 2:
        print(f"  Letter {letter}: Insufficient patterns after filtering "
              f"({len(patterns_above_min)} patterns for selected terms) - skipping")
        return None, None
    
    # Select top N patterns by frequency (from restricted counter)
    selected_patterns = [p for p, _ in restricted_pattern_counter.most_common() 
                         if p in patterns_above_min][:args.top_patterns]
    
    if len(selected_patterns) < 2:
        print(f"  Letter {letter}: Insufficient patterns after top-N selection - skipping")
        return None, None
    
    return selected_terms, selected_patterns


def build_matrix(cooc_counter, terms, patterns):
    """
    Build co-occurrence matrix.
    
    Returns:
        matrix: numpy array of shape (len(terms), len(patterns))
    """
    n_terms = len(terms)
    n_patterns = len(patterns)
    
    matrix = np.zeros((n_terms, n_patterns), dtype=int)
    
    for i, term in enumerate(terms):
        for j, pattern in enumerate(patterns):
            matrix[i, j] = cooc_counter.get((term, pattern), 0)
    
    return matrix


def prune_empty_rows_cols(matrix, terms, patterns, letter):
    """
    Prune empty rows (terms) and columns (patterns) from matrix.
    
    Returns:
        pruned_matrix: matrix with empty rows/cols removed
        pruned_terms: list of terms after pruning
        pruned_patterns: list of patterns after pruning
        stats: dict with pruning statistics
    """
    # Find non-empty columns (patterns with at least one non-zero cell)
    col_sums = matrix.sum(axis=0)
    nonempty_cols = np.where(col_sums > 0)[0]
    
    # Find non-empty rows (terms with at least one non-zero cell)
    row_sums = matrix.sum(axis=1)
    nonempty_rows = np.where(row_sums > 0)[0]
    
    # Prune matrix
    pruned_matrix = matrix[np.ix_(nonempty_rows, nonempty_cols)]
    
    # Prune lists
    pruned_terms = [terms[i] for i in nonempty_rows]
    pruned_patterns = [patterns[j] for j in nonempty_cols]
    
    stats = {
        'original_terms': len(terms),
        'original_patterns': len(patterns),
        'pruned_terms': len(pruned_terms),
        'pruned_patterns': len(pruned_patterns),
        'dropped_terms': len(terms) - len(pruned_terms),
        'dropped_patterns': len(patterns) - len(pruned_patterns)
    }
    
    if stats['dropped_terms'] > 0 or stats['dropped_patterns'] > 0:
        print(f"    Pruning: dropped {stats['dropped_terms']} empty term rows, "
              f"{stats['dropped_patterns']} empty pattern columns")
    
    return pruned_matrix, pruned_terms, pruned_patterns, stats


def apply_cell_threshold(matrix, threshold_ratio, threshold_mode):
    """
    Apply cell threshold: mask cells below threshold.
    
    Args:
        matrix: numpy array
        threshold_ratio: ratio for thresholding
        threshold_mode: 'global', 'per_column', or 'none'
    
    Returns:
        masked matrix
    """
    if matrix.size == 0 or matrix.max() == 0:
        return np.ma.masked_where(matrix == 0, matrix)
    
    if threshold_mode == "none":
        # Only mask zeros, no thresholding
        return np.ma.masked_where(matrix == 0, matrix)
    
    elif threshold_mode == "global":
        # Global threshold: ratio of global max
        max_cell = matrix.max()
        threshold = threshold_ratio * max_cell
        matrix_masked = np.ma.masked_where(matrix < threshold, matrix)
        return matrix_masked
    
    elif threshold_mode == "per_column":
        # Per-column threshold: ratio of each column's max
        matrix_masked = np.ma.copy(matrix)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            max_col = col.max()
            if max_col > 0:
                threshold_col = threshold_ratio * max_col
                matrix_masked[:, j] = np.ma.masked_where(col < threshold_col, col)
        return matrix_masked
    
    else:
        # Default to global if unknown mode
        max_cell = matrix.max()
        threshold = threshold_ratio * max_cell
        return np.ma.masked_where(matrix < threshold, matrix)


def create_heatmap_for_letter(letter, matrix, terms, patterns, out_path, 
                               cell_threshold_ratio, threshold_mode):
    """
    Create and save log-scale heatmap for a single letter.
    """
    # Apply cell threshold (creates masked array)
    matrix_masked = apply_cell_threshold(matrix, cell_threshold_ratio, threshold_mode)
    
    # Further mask zeros for better visualization
    matrix_masked = np.ma.masked_where(matrix_masked == 0, matrix_masked)
    
    if matrix_masked.count() == 0:  # All masked
        print(f"    WARNING: All cells masked/zero after thresholding - skipping plot")
        return False
    
    # Determine figure size
    n_terms = len(terms)
    n_patterns = len(patterns)
    fig_width = max(8, min(20, n_patterns * 0.4))
    fig_height = max(8, min(24, n_terms * 0.3))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create heatmap with log normalization
    vmax = matrix_masked.max()
    norm = LogNorm(vmin=1, vmax=vmax)
    
    im = ax.imshow(matrix_masked, aspect='auto', cmap='viridis', 
                  norm=norm, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(patterns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=8)
    
    # Labels and title
    ax.set_xlabel('Pinyin initial pattern', fontsize=11, fontweight='bold')
    ax.set_ylabel('Chinese term', fontsize=11, fontweight='bold')
    ax.set_title(f"Term frequency conditioned on pinyin initial: {letter}", 
                fontsize=13, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Frequency (log scale)', rotation=270, labelpad=20, fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return True


def print_letter_summary(letter, term_counter, pattern_counter, matrix, terms, patterns, 
                        matrix_before=None, terms_before=None, patterns_before=None):
    """
    Print summary for a single letter with debug stats.
    """
    print(f"\n  Letter: {letter.upper()}")
    print(f"    Total unique terms in bucket: {len(term_counter)}")
    print(f"    Total unique patterns in bucket: {len(pattern_counter)}")
    
    # Show before/after pruning stats if provided
    if matrix_before is not None and terms_before is not None and patterns_before is not None:
        print(f"    Before pruning: {len(terms_before)} terms × {len(patterns_before)} patterns, "
              f"{np.count_nonzero(matrix_before):,} non-zero cells")
        print(f"    After pruning:  {len(terms)} terms × {len(patterns)} patterns, "
              f"{np.count_nonzero(matrix):,} non-zero cells")
    else:
        print(f"    Matrix shape: {len(terms)} terms × {len(patterns)} patterns")
        print(f"    Non-zero cells: {np.count_nonzero(matrix):,} / {matrix.size:,}")
    
    print(f"    Top 5 terms:")
    for rank, (term, count) in enumerate(term_counter.most_common(5), 1):
        print(f"      {rank}. {term:15s} {count:,}")
    
    print(f"    Top 5 patterns:")
    for rank, (pattern, count) in enumerate(pattern_counter.most_common(5), 1):
        print(f"      {rank}. {pattern:10s} {count:,}")


def main():
    args = parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Read and process corpus
    cooc_by_letter, term_total_by_letter, pattern_total_by_letter, stats = \
        read_and_process_corpus(args.csv_path)
    
    if not cooc_by_letter:
        print("ERROR: No valid terms found")
        sys.exit(1)
    
    # Print initial summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Rows read from CSV: {stats['rows_read']:,}")
    print(f"Rows kept after filtering: {stats['rows_kept']:,}")
    print(f"Total CJK jieba terms processed: {stats['total_jieba_terms']:,}")
    print(f"Initial letters with data: {len(cooc_by_letter)}")
    print(f"Letters: {sorted(cooc_by_letter.keys())}")
    print("="*70)
    
    # Determine which letters to plot
    if args.letters:
        letters_to_plot = [l.strip().lower() for l in args.letters.split(',')]
        letters_to_plot = [l for l in letters_to_plot if l in cooc_by_letter]
        if not letters_to_plot:
            print(f"ERROR: None of the specified letters {args.letters} have data")
            sys.exit(1)
        print(f"\nPlotting specified letters: {letters_to_plot}")
    else:
        letters_to_plot = sorted(cooc_by_letter.keys())
        print(f"\nPlotting all {len(letters_to_plot)} letters")
    
    # Set CJK font once before plotting
    set_cjk_font()
    
    # Process each letter
    print(f"\nGenerating heatmaps (threshold_mode={args.threshold_mode})...")
    letters_plotted = 0
    figures_to_show = []
    
    for letter in letters_to_plot:
        cooc_counter = cooc_by_letter[letter]
        term_counter = term_total_by_letter[letter]
        pattern_counter = pattern_total_by_letter[letter]
        
        # Filter and select (with conditioned pattern selection)
        selected_terms, selected_patterns = filter_and_select_for_letter(
            letter, cooc_counter, term_counter, pattern_counter, args
        )
        
        if selected_terms is None:
            continue
        
        # Build matrix
        matrix_before = build_matrix(cooc_counter, selected_terms, selected_patterns)
        terms_before = selected_terms.copy()
        patterns_before = selected_patterns.copy()
        
        # Prune empty rows/columns
        matrix, selected_terms, selected_patterns, prune_stats = prune_empty_rows_cols(
            matrix_before, selected_terms, selected_patterns, letter
        )
        
        # Check if still have enough data after pruning
        if len(selected_terms) < 2 or len(selected_patterns) < 2:
            print(f"    Insufficient data after pruning ({len(selected_terms)} terms, "
                  f"{len(selected_patterns)} patterns) - skipping")
            continue
        
        # Create heatmap
        out_path = out_dir / f"heatmap_letter_{letter}.png"
        success = create_heatmap_for_letter(
            letter, matrix, selected_terms, selected_patterns,
            out_path, args.cell_threshold_ratio, args.threshold_mode
        )
        
        if success:
            letters_plotted += 1
            print(f"  {letter}: Saved to {out_path.name}")
            
            # Print detailed summary with before/after pruning stats
            print_letter_summary(letter, term_counter, pattern_counter, 
                               matrix, selected_terms, selected_patterns,
                               matrix_before, terms_before, patterns_before)
            
            # For interactive display (if requested and not too many)
            if args.show and letters_plotted <= args.max_letters_to_show:
                # Re-create figure for display
                matrix_masked = apply_cell_threshold(matrix, args.cell_threshold_ratio, 
                                                    args.threshold_mode)
                matrix_masked = np.ma.masked_where(matrix_masked == 0, matrix_masked)
                
                n_terms = len(selected_terms)
                n_patterns = len(selected_patterns)
                fig_width = max(8, min(20, n_patterns * 0.4))
                fig_height = max(8, min(24, n_terms * 0.3))
                
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                vmax = matrix_masked.max()
                norm = LogNorm(vmin=1, vmax=vmax)
                im = ax.imshow(matrix_masked, aspect='auto', cmap='viridis', 
                              norm=norm, interpolation='nearest')
                ax.set_xticks(range(len(selected_patterns)))
                ax.set_xticklabels(selected_patterns, rotation=45, ha='right', fontsize=8)
                ax.set_yticks(range(len(selected_terms)))
                ax.set_yticklabels(selected_terms, fontsize=8)
                ax.set_xlabel('Pinyin initial pattern', fontsize=11, fontweight='bold')
                ax.set_ylabel('Chinese term', fontsize=11, fontweight='bold')
                ax.set_title(f"Term frequency conditioned on pinyin initial: {letter}", 
                            fontsize=13, fontweight='bold', pad=15)
                cbar = plt.colorbar(im, ax=ax, pad=0.02)
                cbar.set_label('Frequency (log scale)', rotation=270, labelpad=20, fontsize=10)
                plt.tight_layout()
                figures_to_show.append(fig)
    
    print(f"\n{'='*70}")
    print(f"Generated {letters_plotted} heatmaps")
    print(f"{'='*70}")
    
    # Show figures if requested
    if args.show and figures_to_show:
        if len(figures_to_show) > args.max_letters_to_show:
            print(f"\nShowing first {args.max_letters_to_show} figures only...")
        plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

"""
Publication-quality dot plot for BLiMP results - optimized for poster presentation.

Creates a compact horizontal dot plot showing accuracy across three tokenization
regimes with reference lines and color-blind safe markers.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Poster-optimized style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 10


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load three CSV files and merge into a single sorted DataFrame.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        DataFrame with columns [hanzi, pinyin, initials] sorted by hanzi descending
    """
    data_dir = Path(data_dir)
    
    # Load data
    hanzi = pd.read_csv(data_dir / 'hanzi_results.csv')
    pinyin = pd.read_csv(data_dir / 'pinyin_results.csv')
    initials = pd.read_csv(data_dir / 'initials_results.csv')
    
    # Merge on phenomenon
    merged = hanzi.merge(pinyin, on='phenomenon', suffixes=('_hanzi', '_pinyin'))
    merged = merged.merge(initials, on='phenomenon')
    
    # Rename columns
    merged = merged.rename(columns={
        'accuracy_hanzi': 'hanzi',
        'accuracy_pinyin': 'pinyin',
        'accuracy': 'initials'
    })
    
    # Set index and sort by initials accuracy descending
    merged = merged.set_index('phenomenon')
    merged = merged.sort_values('initials', ascending=False)
    
    return merged[['hanzi', 'pinyin', 'initials']]


def plot_poster_dotplot(df: pd.DataFrame, output_path: str):
    """
    Create a vertical dot plot optimized for poster presentation.
    
    Args:
        df: DataFrame with columns [hanzi, pinyin, initials] indexed by phenomenon
        output_path: Path to save the PDF file
    """
    # Convert to percentages
    df_pct = df * 100
    
    # Color-blind safe palette (based on Wong 2011)
    # Blue, Orange, Sky Blue - all distinguishable with most color blindness types
    colors = {
        'hanzi': '#0072B2',      # Dark blue (filled circle)
        'pinyin': '#E69F00',     # Orange (open circle)
        'initials': '#56B4E9'    # Light blue (triangle)
    }
    
    # Create figure - wide for phenomena on x-axis
    n_phenomena = len(df_pct)
    fig_width = max(20, n_phenomena * 0.25)  # Scale with number of phenomena
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    # X positions for phenomena
    x_positions = range(n_phenomena)
    
    # Plot reference lines at hanzi accuracy (thin, subtle)
    for i, (idx, row) in enumerate(df_pct.iterrows()):
        ax.plot([i - 0.3, i + 0.3], [row['hanzi'], row['hanzi']],
                color='gray', linewidth=0.8, alpha=0.3, zorder=1)
    
    # Plot markers
    # Hanzi: filled circle, darkest
    ax.scatter(x_positions, df_pct['hanzi'],
               marker='o', s=120, color=colors['hanzi'], 
               edgecolors='black', linewidths=1.2,
               label='Hanzi', zorder=3)
    
    # Pinyin: open circle
    ax.scatter(x_positions, df_pct['pinyin'],
               marker='o', s=120, facecolors='none', 
               edgecolors=colors['pinyin'], linewidths=2.5,
               label='Pinyin', zorder=3)
    
    # Initials: triangle, lightest
    ax.scatter(x_positions, df_pct['initials'],
               marker='^', s=120, color=colors['initials'],
               edgecolors='black', linewidths=1.2,
               label='Initials', zorder=3)
    
    # Customize axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_pct.index, rotation=90, ha='center', fontsize=15)
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_xlabel('')  # No x-axis label to save space
    ax.set_ylim(-5, 105)
    ax.set_xlim(-0.5, n_phenomena - 0.5)  # Remove white space at ends
    
    # Add horizontal line at 50% (chance level)
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1.5, alpha=0.4, zorder=2)
    
    # # Add title
    # ax.set_title('BLiMP Accuracy Across Tokenization Regimes', 
    #              fontsize=20, fontweight='bold', pad=20)
    # # Legend above plot (horizontal)1
    # legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02),
    #                   ncol=3, frameon=True, fontsize=16, 
    #                   columnspacing=2.0, handletextpad=0.5)
    # legend.get_frame().set_linewidth(1.5)
    # legend.get_frame().set_edgecolor('black')
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.95)

    # Title
    # ax.set_title(
    #     'BLiMP Accuracy Across Tokenization Regimes',
    #     fontsize=20,
    #     fontweight='bold',
    #     pad=20
    # )

    # Legend placed to the right of the title
    # legend = ax.legend(
    #     loc='upper left',
    #     bbox_to_anchor=(1.02, 1.15),   # move right & slightly above axes
    #     ncol=3,
    #     frameon=True,
    #     fontsize=16,
    #     columnspacing=1.5,
    #     handletextpad=0.5
    # )

    # legend.get_frame().set_linewidth(1.5)
    # legend.get_frame().set_edgecolor('black')
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.95)

    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicker spine lines for poster visibility
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Remove grid
    ax.grid(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved poster dot plot to {output_path}")
    plt.close()


def print_summary(df: pd.DataFrame):
    """
    Print a compact summary of the data.
    
    Args:
        df: DataFrame with columns [hanzi, pinyin, initials]
    """
    print("\n" + "="*60)
    print("POSTER VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Number of phenomena: {len(df)}")
    print(f"\nMean accuracy:")
    print(f"  Hanzi:    {df['hanzi'].mean()*100:.1f}%")
    print(f"  Pinyin:   {df['pinyin'].mean()*100:.1f}%")
    print(f"  Initials: {df['initials'].mean()*100:.1f}%")
    print("\n" + "="*60)


def main():
    """Main execution function."""
    # Set up paths
    data_dir = Path('results/blimp')
    output_path = Path('results/blimp/plots/poster_blimp_compact.pdf')
    
    # Check if CSV files exist
    required_files = ['hanzi_results.csv', 'pinyin_results.csv', 'initials_results.csv']
    if not all((data_dir / f).exists() for f in required_files):
        print("Error: CSV files not found in results/blimp/")
        print("Please run parse_blimp_results.py first.")
        return
    
    # Load and sort data
    print("Loading data...")
    df = load_data(str(data_dir))
    
    # Generate visualization
    print(f"Generating poster visualization for {len(df)} phenomena...")
    plot_poster_dotplot(df, str(output_path))
    
    # Print summary
    print_summary(df)
    
    print("\n[SUCCESS] Poster visualization completed!")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()

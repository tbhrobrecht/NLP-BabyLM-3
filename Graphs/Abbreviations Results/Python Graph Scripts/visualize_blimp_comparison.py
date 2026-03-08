"""
Publication-quality visualization of BLiMP evaluation results across tokenization regimes.

Compares hanzi, pinyin, and pinyin initials tokenization on Chinese linguistic phenomena.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# Publication style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5


def load_and_merge_data(hanzi_csv: str, pinyin_csv: str, initials_csv: str) -> pd.DataFrame:
    """
    Load three CSV files and merge into a single DataFrame indexed by phenomenon.
    
    Args:
        hanzi_csv: Path to hanzi results CSV
        pinyin_csv: Path to pinyin results CSV
        initials_csv: Path to initials results CSV
        
    Returns:
        Merged DataFrame with columns [hanzi, pinyin, initials] indexed by phenomenon
    """
    # Load data
    hanzi = pd.read_csv(hanzi_csv)
    pinyin = pd.read_csv(pinyin_csv)
    initials = pd.read_csv(initials_csv)
    
    # Merge on phenomenon
    merged = hanzi.merge(pinyin, on='phenomenon', suffixes=('_hanzi', '_pinyin'))
    merged = merged.merge(initials, on='phenomenon')
    
    # Rename columns
    merged = merged.rename(columns={
        'accuracy_hanzi': 'hanzi',
        'accuracy_pinyin': 'pinyin',
        'accuracy': 'initials'
    })
    
    # Set phenomenon as index
    merged = merged.set_index('phenomenon')
    
    return merged[['hanzi', 'pinyin', 'initials']]


def plot_grouped_bar_chart(df: pd.DataFrame, output_path: str):
    """
    Create a grouped bar chart comparing accuracy across tokenization regimes.
    
    Args:
        df: DataFrame with columns [hanzi, pinyin, initials] indexed by phenomenon
        output_path: Path to save the PDF file
    """
    # Convert to percentages
    df_pct = df * 100
    
    # Set up color palette (hanzi darkest, initials lightest)
    colors = ['#2c5f8d', '#5c8fb5', '#9dbfd9']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create grouped bars
    x = np.arange(len(df_pct))
    width = 0.25
    
    bars1 = ax.bar(x - width, df_pct['hanzi'], width, label='Hanzi', 
                   color=colors[0], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, df_pct['pinyin'], width, label='Pinyin',
                   color=colors[1], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, df_pct['initials'], width, label='Initials',
                   color=colors[2], edgecolor='black', linewidth=0.5)
    
    # Customize axes
    ax.set_xlabel('Linguistic Phenomenon', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('BLiMP Accuracy Across Tokenization Regimes', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_pct.index, rotation=90, ha='right', fontsize=8)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Add horizontal line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Add value labels on bars (only for bars > 10% for readability)
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 10:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=6, rotation=90)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved grouped bar chart to {output_path}")
    plt.close()


def plot_performance_drop_heatmap(df: pd.DataFrame, output_path: str):
    """
    Create a heatmap showing relative performance drop compared to hanzi.
    
    Args:
        df: DataFrame with columns [hanzi, pinyin, initials] indexed by phenomenon
        output_path: Path to save the PDF file
    """
    # Compute drops vs hanzi
    drop_data = pd.DataFrame({
        'Pinyin Drop': df['hanzi'] - df['pinyin'],
        'Initials Drop': df['hanzi'] - df['initials']
    }, index=df.index)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 18))
    
    # Create heatmap
    sns.heatmap(drop_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                center=0, vmin=-0.5, vmax=0.5,
                cbar_kws={'label': 'Accuracy Drop vs Hanzi'},
                linewidths=0.5, linecolor='gray',
                ax=ax, annot_kws={'fontsize': 7})
    
    # Customize
    ax.set_title('Performance Drop Relative to Hanzi Tokenization',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Tokenization Regime', fontsize=12, fontweight='bold')
    ax.set_ylabel('Linguistic Phenomenon', fontsize=12, fontweight='bold')
    
    # Rotate y-axis labels for better readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def create_summary_statistics(df: pd.DataFrame):
    """
    Print summary statistics for the three tokenization regimes.
    
    Args:
        df: DataFrame with columns [hanzi, pinyin, initials] indexed by phenomenon
    """
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for regime in ['hanzi', 'pinyin', 'initials']:
        print(f"\n{regime.upper()}:")
        print(f"  Mean accuracy: {df[regime].mean():.4f} ({df[regime].mean()*100:.2f}%)")
        print(f"  Std deviation: {df[regime].std():.4f}")
        print(f"  Min accuracy:  {df[regime].min():.4f} ({df[regime].min()*100:.2f}%)")
        print(f"  Max accuracy:  {df[regime].max():.4f} ({df[regime].max()*100:.2f}%)")
    
    # Performance drops
    print(f"\nAVERAGE PERFORMANCE DROP vs HANZI:")
    print(f"  Pinyin:   {(df['hanzi'] - df['pinyin']).mean():.4f} ({(df['hanzi'] - df['pinyin']).mean()*100:.2f}%)")
    print(f"  Initials: {(df['hanzi'] - df['initials']).mean():.4f} ({(df['hanzi'] - df['initials']).mean()*100:.2f}%)")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    # Set up paths
    data_dir = Path('results/blimp')
    plots_dir = Path('results/blimp/plots')
    
    hanzi_csv = data_dir / 'hanzi_results.csv'
    pinyin_csv = data_dir / 'pinyin_results.csv'
    initials_csv = data_dir / 'initials_results.csv'
    
    # Check if CSV files exist
    if not all([hanzi_csv.exists(), pinyin_csv.exists(), initials_csv.exists()]):
        print("Error: CSV files not found. Please run parse_blimp_results.py first.")
        return
    
    # Load and merge data
    print("Loading data...")
    df = load_and_merge_data(str(hanzi_csv), str(pinyin_csv), str(initials_csv))
    print(f"Loaded {len(df)} phenomena")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Grouped bar chart
    bar_chart_path = plots_dir / 'blimp_comparison_bars.pdf'
    plot_grouped_bar_chart(df, str(bar_chart_path))
    
    # Performance drop heatmap
    heatmap_path = plots_dir / 'blimp_performance_drops.pdf'
    plot_performance_drop_heatmap(df, str(heatmap_path))
    
    # Summary statistics
    create_summary_statistics(df)
    
    print("\n[SUCCESS] All visualizations completed successfully!")


if __name__ == '__main__':
    main()

"""
KDE density plot visualization of BLiMP accuracy distributions across tokenization regimes.

Creates overlapping kernel density estimation plots showing the distribution of 
accuracy scores with rug plots for discrete data points.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load three CSV files and merge into a single DataFrame.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        DataFrame with columns [hanzi, pinyin, initials]
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
    
    merged = merged.set_index('phenomenon')
    
    return merged[['hanzi', 'pinyin', 'initials']]


def plot_accuracy_kde(hanzi_acc: np.ndarray, pinyin_acc: np.ndarray, 
                      initials_acc: np.ndarray, output_path: str):
    """
    Create overlapping KDE density plots of accuracy distributions.
    
    Args:
        hanzi_acc: Array of hanzi accuracy values (0-100%)
        pinyin_acc: Array of pinyin accuracy values (0-100%)
        initials_acc: Array of initials accuracy values (0-100%)
        output_path: Path to save the PDF file
    """
    # Color-blind safe palette
    colors = {
        'hanzi': '#0072B2',      # Dark blue
        'pinyin': '#E69F00',     # Orange
        'initials': '#D55E00'    # Vermillion (red-orange)
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot KDE density curves with fill
    sns.kdeplot(data=hanzi_acc, fill=True, alpha=0.25, linewidth=2,
                color=colors['hanzi'], label='Hanzi', clip=(0, 100),
                bw_adjust=1.1, ax=ax)
    
    sns.kdeplot(data=pinyin_acc, fill=True, alpha=0.25, linewidth=2,
                color=colors['pinyin'], label='Pinyin', clip=(0, 100),
                bw_adjust=1.1, ax=ax)
    
    sns.kdeplot(data=initials_acc, fill=True, alpha=0.25, linewidth=2,
                color=colors['initials'], label='Initials', clip=(0, 100),
                bw_adjust=1.1, ax=ax)
    
    # Add rug plots to show discrete data points
    sns.rugplot(data=hanzi_acc, color=colors['hanzi'], alpha=0.3, 
                height=0.03, ax=ax, linewidth=1)
    sns.rugplot(data=pinyin_acc, color=colors['pinyin'], alpha=0.3,
                height=0.03, ax=ax, linewidth=1)
    sns.rugplot(data=initials_acc, color=colors['initials'], alpha=0.3,
                height=0.03, ax=ax, linewidth=1)
    
    # Add vertical line at 50% (chance level)
    # ax.axvline(x=50, color='black', linestyle='--', linewidth=2, 
    #            alpha=0.7, label='Chance (50%)', zorder=10)
    
    # Customize axes
    ax.set_xlabel('Accuracy (%)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Density', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title('Distribution of BLiMP Accuracy Across Tokenization Regimes',
                 fontsize=20, fontweight='bold', pad=20)
    
    # Set limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.014)
    
    # Legend positioned outside to avoid blocking plot
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                      frameon=True, fontsize=14, edgecolor='black', 
                      fancybox=False, framealpha=1.0)
    legend.get_frame().set_linewidth(1.5)
    
    # Light grid for readability
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved KDE density plot to {output_path}")
    plt.close()


def print_distribution_stats(df: pd.DataFrame):
    """
    Print distribution statistics for each regime.
    
    Args:
        df: DataFrame with columns [hanzi, pinyin, initials]
    """
    print("\n" + "="*70)
    print("ACCURACY DISTRIBUTION STATISTICS")
    print("="*70)
    
    for regime in ['hanzi', 'pinyin', 'initials']:
        data = df[regime] * 100
        print(f"\n{regime.upper()}:")
        print(f"  Mean:     {data.mean():.1f}%")
        print(f"  Median:   {data.median():.1f}%")
        print(f"  Std Dev:  {data.std():.1f}%")
        print(f"  Min:      {data.min():.1f}%")
        print(f"  Max:      {data.max():.1f}%")
        print(f"  Below 50%: {(data < 50).sum()} phenomena ({(data < 50).sum()/len(data)*100:.1f}%)")
        print(f"  Above 50%: {(data >= 50).sum()} phenomena ({(data >= 50).sum()/len(data)*100:.1f}%)")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    # Set up paths
    data_dir = Path('results/blimp')
    output_path = Path('results/blimp/plots/poster_blimp_kde.pdf')
    
    # Check if CSV files exist
    required_files = ['hanzi_results.csv', 'pinyin_results.csv', 'initials_results.csv']
    if not all((data_dir / f).exists() for f in required_files):
        print("Error: CSV files not found in results/blimp/")
        print("Please run parse_blimp_results.py first.")
        return
    
    # Load data
    print("Loading data...")
    df = load_data(str(data_dir))
    
    # Convert to percentage arrays
    hanzi_acc = (df['hanzi'] * 100).values
    pinyin_acc = (df['pinyin'] * 100).values
    initials_acc = (df['initials'] * 100).values
    
    # Generate KDE density plot
    print(f"Generating KDE density plot for {len(df)} phenomena...")
    plot_accuracy_kde(hanzi_acc, pinyin_acc, initials_acc, str(output_path))
    
    # Print distribution statistics
    print_distribution_stats(df)
    
    print("\n[SUCCESS] KDE density plot completed!")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()


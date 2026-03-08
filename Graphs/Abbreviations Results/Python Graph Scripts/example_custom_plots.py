"""
Example: Customizing BLiMP visualizations

This script demonstrates common customizations for the visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# Import the base visualization functions
from visualize_blimp_comparison import load_and_merge_data


def plot_subset_comparison(df: pd.DataFrame, output_path: str, subset_pattern: str = None):
    """
    Create bar chart for a subset of phenomena (e.g., only "BA_" phenomena).
    
    Args:
        df: Full DataFrame with all phenomena
        output_path: Path to save the PDF
        subset_pattern: Pattern to filter phenomena (e.g., "BA_", "question_")
    """
    # Filter by pattern if provided
    if subset_pattern:
        df_subset = df[df.index.str.startswith(subset_pattern)]
        title_suffix = f" - {subset_pattern}* Phenomena"
    else:
        df_subset = df
        title_suffix = ""
    
    df_pct = df_subset * 100
    
    # Custom color palette - can be changed
    colors = ['#1f4788', '#2e7d32', '#f57c00']  # Blue, Green, Orange
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(df_pct))
    width = 0.25
    
    bars1 = ax.bar(x - width, df_pct['hanzi'], width, label='Hanzi',
                   color=colors[0], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, df_pct['pinyin'], width, label='Pinyin',
                   color=colors[1], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, df_pct['initials'], width, label='Initials',
                   color=colors[2], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Linguistic Phenomenon', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'BLiMP Accuracy Comparison{title_suffix}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_pct.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved subset comparison to {output_path}")
    plt.close()


def plot_difference_chart(df: pd.DataFrame, output_path: str):
    """
    Create a chart showing the difference (pinyin - hanzi) and (initials - hanzi).
    Useful for seeing which phenomena benefit from alternative tokenization.
    
    Args:
        df: DataFrame with columns [hanzi, pinyin, initials]
        output_path: Path to save the PDF
    """
    differences = pd.DataFrame({
        'Pinyin Gain': (df['pinyin'] - df['hanzi']) * 100,
        'Initials Gain': (df['initials'] - df['hanzi']) * 100
    }, index=df.index)
    
    # Sort by pinyin gain for better visualization
    differences = differences.sort_values('Pinyin Gain')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(differences))
    width = 0.35
    
    bars1 = ax.barh(x, differences['Pinyin Gain'], width, label='Pinyin vs Hanzi',
                    color=['#d32f2f' if v < 0 else '#388e3c' for v in differences['Pinyin Gain']],
                    edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(x + width, differences['Initials Gain'], width, label='Initials vs Hanzi',
                    color=['#d32f2f' if v < 0 else '#388e3c' for v in differences['Initials Gain']],
                    edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Linguistic Phenomenon', fontsize=12, fontweight='bold')
    ax.set_xlabel('Accuracy Gain/Loss (%)', fontsize=12, fontweight='bold')
    ax.set_title('Tokenization Impact: Change from Hanzi Baseline',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks(x + width/2)
    ax.set_yticklabels(differences.index, fontsize=7)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved difference chart to {output_path}")
    plt.close()


def plot_scatter_comparison(df: pd.DataFrame, output_path: str):
    """
    Create scatter plots comparing regimes pairwise.
    
    Args:
        df: DataFrame with columns [hanzi, pinyin, initials]
        output_path: Path to save the PDF
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    comparisons = [
        ('hanzi', 'pinyin', 'Hanzi vs Pinyin'),
        ('hanzi', 'initials', 'Hanzi vs Initials'),
        ('pinyin', 'initials', 'Pinyin vs Initials')
    ]
    
    for ax, (x_col, y_col, title) in zip(axes, comparisons):
        ax.scatter(df[x_col] * 100, df[y_col] * 100, alpha=0.6, s=50)
        ax.plot([0, 100], [0, 100], 'r--', linewidth=1, label='y=x (equal performance)')
        
        ax.set_xlabel(f'{x_col.capitalize()} Accuracy (%)', fontsize=11)
        ax.set_ylabel(f'{y_col.capitalize()} Accuracy (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved scatter comparison to {output_path}")
    plt.close()


def main():
    """Generate additional custom visualizations."""
    # Load data
    data_dir = Path('results/blimp')
    plots_dir = Path('results/blimp/plots/custom')
    
    df = load_and_merge_data(
        str(data_dir / 'hanzi_results.csv'),
        str(data_dir / 'pinyin_results.csv'),
        str(data_dir / 'initials_results.csv')
    )
    
    print(f"Loaded {len(df)} phenomena\n")
    
    # Generate custom plots
    print("Generating custom visualizations...")
    
    # 1. BA construction subset
    plot_subset_comparison(df, str(plots_dir / 'ba_constructions.pdf'), subset_pattern='BA_')
    
    # 2. Question phenomena subset
    plot_subset_comparison(df, str(plots_dir / 'question_phenomena.pdf'), subset_pattern='question_')
    
    # 3. Difference chart (gain/loss)
    plot_difference_chart(df, str(plots_dir / 'tokenization_impact.pdf'))
    
    # 4. Scatter comparison
    plot_scatter_comparison(df, str(plots_dir / 'pairwise_scatter.pdf'))
    
    print("\n[SUCCESS] All custom visualizations completed!")
    print(f"\nGenerated files in {plots_dir}/:")
    print("  - ba_constructions.pdf")
    print("  - question_phenomena.pdf")
    print("  - tokenization_impact.pdf")
    print("  - pairwise_scatter.pdf")


if __name__ == '__main__':
    main()

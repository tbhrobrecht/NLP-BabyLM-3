"""Parse blimp_results file and create CSV files for each tokenization regime."""

import re
import pandas as pd
from pathlib import Path


def parse_blimp_results(results_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse blimp_results file and extract data for three tokenization regimes.
    
    Args:
        results_file: Path to the blimp_results text file
        
    Returns:
        Tuple of (hanzi_df, pinyin_df, initials_df)
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract PINYIN INITIALS section
    initials_section = re.search(
        r'RESULTS PINYIN INITIALS.*?\n-{80}\n(.*?)\n-{80}',
        content, re.DOTALL
    )
    
    # Extract HANZI section
    hanzi_section = re.search(
        r'Results Summary: HANZI.*?\n={60}\n(.*?)\n={60}',
        content, re.DOTALL
    )
    
    # Extract PINYIN section
    pinyin_section = re.search(
        r'Results Summary: PINYIN.*?\n={60}\n(.*?)\n={60}',
        content, re.DOTALL
    )
    
    # Parse initials data
    initials_data = []
    if initials_section:
        lines = initials_section.group(1).strip().split('\n')
        for line in lines:
            # Match lines like: "BA_deletion                                         95.00%       285/300"
            match = re.match(r'\s*(\S+(?:_\S+)*)\s+(\d+\.\d+)%', line)
            if match:
                phenomenon = match.group(1)
                accuracy = float(match.group(2)) / 100.0
                initials_data.append({'phenomenon': phenomenon, 'accuracy': accuracy})
    
    # Parse hanzi data
    hanzi_data = []
    if hanzi_section:
        lines = hanzi_section.group(1).strip().split('\n')
        for line in lines:
            # Match lines like: "  BA_BEI_subj_drop: 0.4667"
            match = re.match(r'\s+(\S+(?:_\S+)*):\s+(\d+\.\d+)', line)
            if match:
                phenomenon = match.group(1)
                accuracy = float(match.group(2))
                hanzi_data.append({'phenomenon': phenomenon, 'accuracy': accuracy})
    
    # Parse pinyin data
    pinyin_data = []
    if pinyin_section:
        lines = pinyin_section.group(1).strip().split('\n')
        for line in lines:
            # Match lines like: "  BA_BEI_subj_drop: 0.7400"
            match = re.match(r'\s+(\S+(?:_\S+)*):\s+(\d+\.\d+)', line)
            if match:
                phenomenon = match.group(1)
                accuracy = float(match.group(2))
                pinyin_data.append({'phenomenon': phenomenon, 'accuracy': accuracy})
    
    return (
        pd.DataFrame(hanzi_data),
        pd.DataFrame(pinyin_data),
        pd.DataFrame(initials_data)
    )


def main():
    """Parse blimp_results and save as CSV files."""
    # Set up paths
    results_file = Path('blimp_results')
    output_dir = Path('results/blimp')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse results
    print("Parsing blimp_results file...")
    hanzi_df, pinyin_df, initials_df = parse_blimp_results(results_file)
    
    # Save to CSV
    hanzi_df.to_csv(output_dir / 'hanzi_results.csv', index=False)
    pinyin_df.to_csv(output_dir / 'pinyin_results.csv', index=False)
    initials_df.to_csv(output_dir / 'initials_results.csv', index=False)
    
    print(f"Saved results to {output_dir}/")
    print(f"  - hanzi_results.csv: {len(hanzi_df)} phenomena")
    print(f"  - pinyin_results.csv: {len(pinyin_df)} phenomena")
    print(f"  - initials_results.csv: {len(initials_df)} phenomena")


if __name__ == '__main__':
    main()

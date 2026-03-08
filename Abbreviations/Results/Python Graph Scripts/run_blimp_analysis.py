#!/usr/bin/env python3
"""
Master script to parse BLiMP results and generate all visualizations.

Usage:
    python run_blimp_analysis.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: list, description: str) -> bool:
    """
    Run a command and report status.
    
    Args:
        command: Command to run as list of strings
        description: Human-readable description
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """Run the complete BLiMP analysis pipeline."""
    print("\n" + "="*70)
    print("BLiMP TOKENIZATION COMPARISON ANALYSIS")
    print("="*70)
    
    # Get Python executable
    python_exe = sys.executable
    
    # Step 1: Parse results
    success = run_command(
        [python_exe, 'parse_blimp_results.py'],
        "Parsing blimp_results file"
    )
    
    if not success:
        print("\n[ERROR] Pipeline failed at parsing step")
        return 1
    
    # Step 2: Generate visualizations
    success = run_command(
        [python_exe, 'visualize_blimp_comparison.py'],
        "Generating visualizations"
    )
    
    if not success:
        print("\n[ERROR] Pipeline failed at visualization step")
        return 1
    
    # Success!
    print("\n" + "="*70)
    print("[SUCCESS] COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/blimp/hanzi_results.csv")
    print("  - results/blimp/pinyin_results.csv")
    print("  - results/blimp/initials_results.csv")
    print("  - results/blimp/plots/blimp_comparison_bars.pdf")
    print("  - results/blimp/plots/blimp_performance_drops.pdf")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

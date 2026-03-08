#!/usr/bin/env python3
"""
Convert BLiMP Chinese .jsonl files from Hanzi to jieba-segmented pinyin with tone marks.

Each jieba token becomes ONE pinyin token (syllables concatenated, no spaces inside).
Preserves JSON structure: only modifies sentence_good and sentence_bad values.

Example:
    Input:  {"sentence_good": "王姨的老师没有来。", "sentence_bad": "王姨的老虎没有来。"}
    Output: {"sentence_good": "wángyí de lǎoshī méiyǒu lái 。", "sentence_bad": "wángyí de lǎohǔ méiyǒu lái 。"}
"""

import argparse
import json
import logging
import re
import string
from pathlib import Path
from typing import Dict, Tuple

import jieba
from pypinyin import pinyin, Style


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_punct(token: str) -> bool:
    """
    Check if token contains only punctuation characters.
    
    Args:
        token: Input token string
        
    Returns:
        True if token is punctuation-only, False otherwise
    """
    if not token:
        return False
    # Chinese and English punctuation
    punct_chars = set(string.punctuation + '。！？：，、；（）""《》''【】…—·')
    return all(c in punct_chars or c.isspace() for c in token)


def is_ascii_alnum(token: str) -> bool:
    """
    Check if token contains only ASCII letters and/or numbers.
    
    Args:
        token: Input token string
        
    Returns:
        True if token is ASCII alphanumeric only, False otherwise
    """
    if not token:
        return False
    return all(c.isascii() and (c.isalnum() or c in '-_') for c in token)


def jieba_token_to_pinyin_joined(token: str) -> str:
    """
    Convert a single jieba token to pinyin with syllables joined (no spaces inside).
    
    For punctuation or ASCII alphanumeric tokens, returns unchanged.
    For Chinese characters, converts to pinyin with tone marks and joins syllables.
    
    Args:
        token: Single jieba-segmented token
        
    Returns:
        Pinyin representation with syllables joined, or original token if punct/ASCII
        
    Example:
        "老师" -> "lǎoshī"
        "没有" -> "méiyǒu"
        "。" -> "。"
        "iPhone" -> "iPhone"
    """
    token = token.strip()
    
    if not token:
        return token
    
    # Keep punctuation unchanged
    if is_punct(token):
        return token
    
    # Keep ASCII alphanumeric unchanged
    if is_ascii_alnum(token):
        return token
    
    # Convert to pinyin with tone marks
    # pypinyin.pinyin() returns list of lists: [['lǎo'], ['shī']]
    # We need to flatten and join: 'lǎoshī'
    syllables_list = pinyin(token, style=Style.TONE)
    
    # Flatten the list of lists and join syllables without spaces
    syllables = [syl[0] for syl in syllables_list]
    joined = ''.join(syllables)
    
    return joined


def hanzi_to_jieba_pinyin_joined(sentence: str) -> str:
    """
    Convert Chinese sentence to jieba-segmented pinyin with tone marks.
    Each jieba token's syllables are joined (no internal spaces).
    
    Args:
        sentence: Input Chinese sentence
        
    Returns:
        Space-separated pinyin tokens, one per jieba segment
        
    Example:
        "王姨的老师没有来。" -> "wángyí de lǎoshī méiyǒu lái 。"
    """
    if not sentence or not sentence.strip():
        return sentence
    
    # Segment using jieba
    tokens = jieba.lcut(sentence)
    
    # Convert each token to pinyin (joined syllables)
    pinyin_tokens = [jieba_token_to_pinyin_joined(token) for token in tokens]
    
    # Join with single spaces
    result = ' '.join(pinyin_tokens)
    
    # Normalize whitespace: collapse multiple spaces, strip ends
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def convert_record(obj: Dict) -> Dict:
    """
    Convert a single JSON record by transforming sentence_good and sentence_bad.
    
    Modifies the input dictionary in-place and returns it.
    Only converts the two sentence fields; preserves all other keys/values.
    
    Args:
        obj: JSON object (dict) from .jsonl line
        
    Returns:
        Same dictionary with sentence_good and sentence_bad converted to pinyin
        
    Raises:
        KeyError: If required keys are missing
        TypeError: If sentence values are not strings
    """
    # Validate required keys
    if 'sentence_good' not in obj:
        raise KeyError("Missing 'sentence_good' key")
    if 'sentence_bad' not in obj:
        raise KeyError("Missing 'sentence_bad' key")
    
    # Validate types
    if not isinstance(obj['sentence_good'], str):
        raise TypeError(f"'sentence_good' must be string, got {type(obj['sentence_good'])}")
    if not isinstance(obj['sentence_bad'], str):
        raise TypeError(f"'sentence_bad' must be string, got {type(obj['sentence_bad'])}")
    
    # Convert sentences
    obj['sentence_good'] = hanzi_to_jieba_pinyin_joined(obj['sentence_good'])
    obj['sentence_bad'] = hanzi_to_jieba_pinyin_joined(obj['sentence_bad'])
    
    return obj


def convert_file(in_path: Path, out_path: Path) -> Tuple[int, int]:
    """
    Convert a single .jsonl file from Hanzi to pinyin.
    
    Processes line-by-line, streaming through the file.
    Skips invalid lines with warnings.
    
    Args:
        in_path: Input .jsonl file path
        out_path: Output .jsonl file path
        
    Returns:
        Tuple of (lines_written, lines_skipped)
    """
    lines_written = 0
    lines_skipped = 0
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(in_path, 'r', encoding='utf-8') as f_in, \
             open(out_path, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, start=1):
                line = line.strip()
                
                if not line:
                    # Skip empty lines
                    continue
                
                try:
                    # Parse JSON
                    obj = json.loads(line)
                    
                    # Convert the record
                    converted_obj = convert_record(obj)
                    
                    # Write to output (one JSON per line, no ASCII escaping)
                    json_str = json.dumps(converted_obj, ensure_ascii=False)
                    f_out.write(json_str + '\n')
                    
                    lines_written += 1
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error in {in_path}:{line_num} - {e}")
                    lines_skipped += 1
                    
                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid record in {in_path}:{line_num} - {e}")
                    lines_skipped += 1
                    
                except Exception as e:
                    logger.warning(f"Unexpected error in {in_path}:{line_num} - {e}")
                    lines_skipped += 1
    
    except Exception as e:
        logger.error(f"Failed to process file {in_path}: {e}")
        raise
    
    return lines_written, lines_skipped


def main() -> None:
    """
    Main entry point for the conversion script.
    
    Parses arguments, finds all .jsonl files in input directory,
    converts them preserving folder structure, and reports summary.
    """
    parser = argparse.ArgumentParser(
        description='Convert BLiMP Chinese .jsonl files from Hanzi to jieba-segmented pinyin with tone marks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python convert_blimp_hanzi_to_jieba_pinyin_tone_joined_jsonl.py \\
        --input_dir blimp_data \\
        --output_dir outputs/blimp_jieba_pinyin_tone_joined

    python convert_blimp_hanzi_to_jieba_pinyin_tone_joined_jsonl.py \\
        --input_dir blimp_data \\
        --output_dir outputs/blimp_jieba_pinyin_tone_joined \\
        --overwrite
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=Path,
        required=True,
        help='Input directory containing .jsonl files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Output directory for converted .jsonl files'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    if not args.input_dir.is_dir():
        logger.error(f"Input path is not a directory: {args.input_dir}")
        return
    
    # Find all .jsonl files recursively
    jsonl_files = list(args.input_dir.rglob('*.jsonl'))
    
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(jsonl_files)} .jsonl file(s) to process")
    
    # Process each file
    total_files_processed = 0
    total_files_skipped = 0
    total_lines_written = 0
    total_lines_skipped = 0
    
    for in_path in jsonl_files:
        # Compute relative path and output path
        rel_path = in_path.relative_to(args.input_dir)
        out_path = args.output_dir / rel_path
        
        # Check if output exists and overwrite flag
        if out_path.exists() and not args.overwrite:
            logger.info(f"Skipping {rel_path} (output exists, use --overwrite to replace)")
            total_files_skipped += 1
            continue
        
        logger.info(f"Processing {rel_path}...")
        
        try:
            lines_written, lines_skipped = convert_file(in_path, out_path)
            
            total_files_processed += 1
            total_lines_written += lines_written
            total_lines_skipped += lines_skipped
            
            logger.info(f"  ✓ {lines_written} lines written, {lines_skipped} lines skipped")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to process {rel_path}: {e}")
            total_files_skipped += 1
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files processed:     {total_files_processed}")
    logger.info(f"Files skipped:       {total_files_skipped}")
    logger.info(f"Total lines written: {total_lines_written}")
    logger.info(f"Total lines skipped: {total_lines_skipped}")
    logger.info("=" * 60)
    
    if total_files_processed > 0:
        logger.info(f"✓ Conversion complete! Output written to: {args.output_dir}")
    else:
        logger.warning("No files were processed")


if __name__ == '__main__':
    main()

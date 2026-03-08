#!/usr/bin/env python3
"""
count_sentence_bytes.py

Measures byte lengths of sentences and produces jieba-segmented pinyin 
and BabyLM "initials" format for sentences containing Hanzi.

BabyLM format conventions:
- Lowercase a-z initials with single spaces between word-level tokens
- Each token represents one jieba word's concatenated syllable initials
- Example: "已经很晚了" → jieba ["已经","很","晚了"] → "yj h wl"

Usage examples:
    # Single sentence with Hanzi
    python count_sentence_bytes.py --text "已经很晚了"
    # Output includes: babylm_initials = "yj h wl"
    
    # English sentence (no Hanzi processing)
    python count_sentence_bytes.py --text "hello world"
    # Output includes only basic byte stats
    
    # Process file (one sentence per line)
    python count_sentence_bytes.py --file input.txt --out results.jsonl
    
    # Pretty print for single sentence
    python count_sentence_bytes.py --text "测试" --pretty
"""

import argparse
import json
import sys
from typing import Dict, List, Any


def check_dependencies():
    """Check required dependencies and provide helpful error messages."""
    missing = []
    
    try:
        import jieba
    except ImportError:
        missing.append("jieba")
    
    try:
        import pypinyin
    except ImportError:
        missing.append("pypinyin")
    
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}", file=sys.stderr)
        print(f"\nInstall with:", file=sys.stderr)
        print(f"  pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)


check_dependencies()

import jieba
from pypinyin import lazy_pinyin, Style


def contains_hanzi(text: str) -> bool:
    """
    Check if text contains any Hanzi (CJK Unified Ideographs).
    
    Args:
        text: Input string
        
    Returns:
        True if at least one Hanzi character is present
    """
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def get_byte_len(text: str, encoding: str = 'utf-8') -> int:
    """
    Get byte length of text in specified encoding.
    
    Args:
        text: Input string
        encoding: Character encoding (default: utf-8)
        
    Returns:
        Number of bytes
    """
    return len(text.encode(encoding))


def jieba_segment(text: str) -> List[str]:
    """
    Segment text using jieba.
    
    Args:
        text: Input string
        
    Returns:
        List of segmented tokens
    """
    return jieba.lcut(text)


def extract_hanzi_only(text: str) -> str:
    """
    Extract only Hanzi characters from text.
    
    Args:
        text: Input string potentially containing mixed content
        
    Returns:
        String containing only Hanzi characters
    """
    return ''.join(char for char in text if '\u4e00' <= char <= '\u9fff')


def token_to_pinyin_syllables(token: str) -> List[str]:
    """
    Convert a token to pinyin syllables, extracting only Hanzi first.
    
    Args:
        token: Input token (may contain mixed Hanzi and non-Hanzi)
        
    Returns:
        List of pinyin syllables (lowercase, no tones)
    """
    hanzi_only = extract_hanzi_only(token)
    if not hanzi_only:
        return []
    
    # Use pypinyin with NORMAL style (no tones)
    syllables = lazy_pinyin(hanzi_only, style=Style.NORMAL)
    return syllables


def syllables_to_initials(syllables: List[str]) -> str:
    """
    Convert pinyin syllables to concatenated initials.
    
    Args:
        syllables: List of pinyin syllables (e.g., ["yi", "jing"])
        
    Returns:
        Concatenated initials string (e.g., "yj")
    """
    if not syllables:
        return ""
    
    initials = []
    for syl in syllables:
        if syl and syl[0].isalpha():
            initials.append(syl[0].lower())
    
    return ''.join(initials)


def hanzi_to_project_formats(text: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Convert Hanzi text to all project-specific formats.
    
    Args:
        text: Input text containing Hanzi
        encoding: Character encoding for byte counting
        
    Returns:
        Dictionary with all Hanzi-specific fields:
        - hanzi: original text
        - jieba_words: list of jieba tokens
        - jieba_segmented_hanzi: space-joined jieba tokens
        - pinyin_full_by_word: list of pinyin strings per jieba word
        - pinyin_full_words: space-joined pinyin (with word boundaries preserved)
        - pinyin_initials_by_word: list of initial strings per jieba word
        - babylm_initials: space-joined initials (BabyLM format)
        - byte_len_hanzi: byte length of original
        - byte_len_jieba_segmented_hanzi: byte length of segmented
        - byte_len_pinyin_full: byte length of full pinyin
        - byte_len_babylm: byte length of BabyLM format
    """
    # Jieba segmentation
    jieba_words = jieba_segment(text)
    jieba_segmented = ' '.join(jieba_words)
    
    # Process each jieba word for pinyin and initials
    pinyin_full_by_word = []
    pinyin_initials_by_word = []
    
    for word in jieba_words:
        syllables = token_to_pinyin_syllables(word)
        if syllables:
            # Full pinyin for this word: syllables joined by spaces
            pinyin_word = ' '.join(syllables)
            pinyin_full_by_word.append(pinyin_word)
            
            # Initials for this word: concatenated first letters
            initials_word = syllables_to_initials(syllables)
            pinyin_initials_by_word.append(initials_word)
    
    # Join all words with spaces (preserving word boundaries)
    pinyin_full_words = ' '.join(pinyin_full_by_word)
    babylm_initials = ' '.join(pinyin_initials_by_word)
    
    return {
        'hanzi': text, 
        'jieba_words': jieba_words,
        'jieba_segmented_hanzi': jieba_segmented,
        'pinyin_full_by_word': pinyin_full_by_word,
        'pinyin_full_words': pinyin_full_words,
        'pinyin_initials_by_word': pinyin_initials_by_word,
        'babylm_initials': babylm_initials,
        'byte_len_hanzi': get_byte_len(text, encoding),
        'byte_len_jieba_segmented_hanzi': get_byte_len(jieba_segmented, encoding),
        'byte_len_pinyin_full': get_byte_len(pinyin_full_words, encoding),
        'byte_len_babylm': get_byte_len(babylm_initials, encoding),
    }


def process_sentence(text: str, encoding: str = 'utf-8', verbose: bool = False) -> Dict[str, Any]:
    """
    Process a single sentence and return all computed fields.
    
    Args:
        text: Input sentence (not stripped)
        encoding: Character encoding for byte counting
        verbose: If True, print debug information
        
    Returns:
        Dictionary with all computed fields
    """
    result = {
        'text': text,
        'has_hanzi': contains_hanzi(text),
        'char_len': len(text),
        'byte_len': get_byte_len(text, encoding),
        'encoding': encoding,
    }
    
    if result['has_hanzi']:
        # Add all Hanzi-specific fields
        hanzi_data = hanzi_to_project_formats(text, encoding)
        result.update(hanzi_data)
        
        if verbose:
            print(f"[DEBUG] Jieba segmentation: {hanzi_data['jieba_words']}", file=sys.stderr)
            print(f"[DEBUG] Pinyin by word: {hanzi_data['pinyin_full_by_word']}", file=sys.stderr)
            print(f"[DEBUG] Initials by word: {hanzi_data['pinyin_initials_by_word']}", file=sys.stderr)
            print(f"[DEBUG] BabyLM format: {hanzi_data['babylm_initials']}", file=sys.stderr)
    
    return result


def process_text_input(text: str, encoding: str, pretty: bool, verbose: bool):
    """Process single text input and print JSON to stdout."""
    result = process_sentence(text, encoding, verbose)
    
    if pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


def process_file_input(input_path: str, output_path: str, encoding: str, verbose: bool):
    """Process file input (one sentence per line) and write JSONL."""
    try:
        with open(input_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"Error: Failed to decode file with encoding '{encoding}': {e}", file=sys.stderr)
        sys.exit(1)
    
    results = []
    hanzi_count = 0
    
    for line_num, line in enumerate(lines, 1):
        # Remove only trailing newline, preserve all other whitespace
        text = line.rstrip('\n')
        
        if verbose:
            print(f"[DEBUG] Processing line {line_num}: {repr(text)}", file=sys.stderr)
        
        result = process_sentence(text, encoding, verbose)
        results.append(result)
        
        if result['has_hanzi']:
            hanzi_count += 1
    
    # Write output
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Wrote {len(results)} records to {output_path}", file=sys.stderr)
        except IOError as e:
            print(f"Error: Failed to write output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Write to stdout
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
    
    # Summary to stderr
    print(f"\nProcessing summary:", file=sys.stderr)
    print(f"  Total lines: {len(results)}", file=sys.stderr)
    print(f"  Lines with Hanzi: {hanzi_count}", file=sys.stderr)
    print(f"  Lines without Hanzi: {len(results) - hanzi_count}", file=sys.stderr)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Measure byte lengths and produce BabyLM initials format for sentences.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text',
        type=str,
        help='Single sentence to process'
    )
    input_group.add_argument(
        '--file',
        type=str,
        help='Input file with one sentence per line'
    )
    
    # Optional arguments
    parser.add_argument(
        '--encoding',
        type=str,
        default='utf-8',
        help='Character encoding for byte counting and file I/O (default: utf-8)'
    )
    parser.add_argument(
        '--out',
        type=str,
        help='Output file path for JSONL (only for --file mode)'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON output (only for --text mode)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debug information to stderr'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.out and not args.file:
        parser.error("--out can only be used with --file")
    
    # Process input
    if args.text:
        process_text_input(args.text, args.encoding, args.pretty, args.verbose)
    else:
        process_file_input(args.file, args.out, args.encoding, args.verbose)


if __name__ == '__main__':
    main()

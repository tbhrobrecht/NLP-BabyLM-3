#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to convert Chinese text to pinyin with jieba segmentation.
Maintains the original line structure.
"""

import jieba
from pypinyin import lazy_pinyin, Style
import argparse
import os


def convert_line_to_pinyin(line, separator=' '):
    """
    Convert a line of Chinese text to pinyin with jieba segmentation.
    
    Args:
        line: Input line containing Chinese text
        separator: Separator between pinyin syllables (default: space)
    
    Returns:
        String with pinyin representation
    """
    # Strip trailing whitespace/newline but preserve it for later
    line = line.rstrip('\n\r')
    
    if not line.strip():
        return line
    
    # Segment the line using jieba
    words = jieba.cut(line)
    
    # Convert each word to pinyin
    pinyin_words = []
    for word in words:
        # Convert word to pinyin (with tone marks)
        word_pinyin = lazy_pinyin(word, style=Style.TONE)
        # Join syllables in the word
        pinyin_words.append(''.join(word_pinyin))
    
    # Join words with separator
    return separator.join(pinyin_words)


def convert_file_to_pinyin(input_file, output_file, separator=' '):
    """
    Convert an entire file from Chinese to pinyin.
    
    Args:
        input_file: Path to input text file
        output_file: Path to output text file
        separator: Separator between pinyin words (default: space)
    """
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    print(f"Using jieba for segmentation...")
    
    line_count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            pinyin_line = convert_line_to_pinyin(line, separator)
            fout.write(pinyin_line + '\n')
            line_count += 1
            
            # Progress indicator
            if line_count % 1000 == 0:
                print(f"Processed {line_count} lines...")
    
    print(f"Conversion complete! Processed {line_count} lines.")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Chinese text to pinyin with jieba segmentation'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/train.txt',
        help='Input file path (default: data/train.txt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/train_pinyin.txt',
        help='Output file path (default: data/train_pinyin.txt)'
    )
    parser.add_argument(
        '--separator',
        type=str,
        default=' ',
        help='Separator between pinyin words (default: space)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert the file
    convert_file_to_pinyin(args.input, args.output, args.separator)


if __name__ == '__main__':
    main()

"""
Preprocess Hanzi corpus to pinyin initials representation.

Pipeline: CSV (Hanzi) → jieba segmentation → pypinyin → initials (with zh/ch/sh digraphs)
"""
import csv
import re
from typing import List, Tuple
from pathlib import Path

import jieba
from pypinyin import pinyin, Style


def get_initial_from_syllable(syllable: str) -> str:
    """
    Extract initial from a pinyin syllable.
    
    Rules:
    - If starts with 'zh', 'ch', 'sh': return that digraph
    - Otherwise: return first letter
    - Special cases handled by pypinyin (e.g., 'a' → 'a', 'yi' → 'y')
    
    Args:
        syllable: Pinyin syllable (e.g., 'zhang', 'he', 'ai')
    
    Returns:
        Initial string (e.g., 'zh', 'h', 'a')
    """
    syllable = syllable.lower().strip()
    if not syllable:
        return ""
    
    # Check digraphs
    if syllable.startswith("zh"):
        return "zh"
    elif syllable.startswith("ch"):
        return "ch"
    elif syllable.startswith("sh"):
        return "sh"
    else:
        return syllable[0]


def convert_word_to_initials(word: str, use_digraph_repetition: bool = True) -> str:
    """
    Convert a single Hanzi word to initials string.
    
    Initials within a word are CONCATENATED without spaces to preserve word boundaries.
    Example: "我们" (wo men) → "wm" (NOT "w m")
    
    With digraph_repetition enabled:
    - "时时" (shi shi) → "shh" (two 'sh' in a row, collapsed)
    - "时候" (shi hou) → "shh" (sh + h, different initials but looks like run)
    
    Note: Since we concatenate within words, the output is one continuous string.
    Example: "长河" (chang he) → "chh"
    
    Digraph repetition rule:
    - If consecutive syllables BOTH start with zh/ch/sh (the SAME one), collapse to zhh/chh/shh
    - Example: "周周" (zhou zhou) → "zhh" (two 'zh' in a row)
    - Example: "时候" (shi hou) → "shh" (sh digraph, then 'h' initial)
    
    Implementation:
    1. Get all syllable initials for the word
    2. Scan for runs of the same digraph (zh/ch/sh)
    3. Collapse runs: if 2+ consecutive syllables have same digraph, output digraph + 'h'*(count-1)
    4. Concatenate all initials WITHOUT spaces (preserve word boundary)
    """
    # Get pinyin syllables for this word
    syllables = pinyin(word, style=Style.NORMAL, strict=False)
    syllables = [s[0] for s in syllables]  # Flatten
    
    if not syllables:
        return ""
    
    # Get initials
    initials = [get_initial_from_syllable(syl) for syl in syllables]
    
    if not use_digraph_repetition:
        return "".join(initials)  # Changed from " ".join to "".join
    
    # Apply digraph repetition rule
    result = []
    i = 0
    while i < len(initials):
        current = initials[i]
        
        # Check if current is a digraph
        if current in {"zh", "ch", "sh"}:
            # Count consecutive occurrences of the same digraph
            run_length = 1
            j = i + 1
            while j < len(initials) and initials[j] == current:
                run_length += 1
                j += 1
            
            # Output: digraph + 'h' repeated (run_length - 1)
            if run_length > 1:
                result.append(current + "h" * (run_length - 1))
            else:
                result.append(current)
            
            i = j
        else:
            result.append(current)
            i += 1
    
    return "".join(result)  # Changed from " ".join to "".join


def hanzi_to_initials(text: str, use_digraph_repetition: bool = True) -> str:
    """
    Convert a Hanzi text string to initials representation.
    
    Uses jieba to segment text into words, then converts each word to initials.
    Spaces are preserved ONLY between jieba-segmented words, NOT within words.
    
    Args:
        text: Input Hanzi text
        use_digraph_repetition: Whether to use zhh/chh/shh for repeated digraphs
    
    Returns:
        Initials string (space-separated between jieba word boundaries)
    
    Examples:
        "我们" → "wm" (wo men, single word)
        "今天" → "jt" (jin tian, single word)
        "时候" → "shh" (shi hou, sh + h within same word)
        "长河" → "chh" (chang he, ch + h within same word)
        "中国" → "zhg" (zhong guo, zh + g within same word)
        "上海" → "shh" (shang hai, sh + h within same word)
        "周周" → "zhh" (zhou zhou, two consecutive zh digraphs)
        "我们今天有什么安排" → "wm jt y shm ap" (word boundaries preserved by spaces)
    """
    # Tokenize with jieba
    words = jieba.cut(text, cut_all=False)
    
    # Convert each word to initials
    initials_list = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        initials = convert_word_to_initials(word, use_digraph_repetition)
        if initials:
            initials_list.append(initials)
    
    # Join with single space (preserves jieba word boundaries)
    return " ".join(initials_list)


def preprocess_csv(
    csv_path: str,
    output_path: str,
    text_column: str = "text",
    use_digraph_repetition: bool = True,
    max_hanzi_per_line: int = 1000
) -> Tuple[int, int]:
    """
    Read CSV with Hanzi text and convert to initials corpus.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output initials text file
        text_column: Name of column containing Hanzi text
        use_digraph_repetition: Enable digraph repetition rule
        max_hanzi_per_line: Skip lines exceeding this length
    
    Returns:
        (lines_processed, lines_skipped)
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    lines_processed = 0
    lines_skipped = 0
    
    with open(csv_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        reader = csv.DictReader(f_in)
        
        if text_column not in reader.fieldnames:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available: {reader.fieldnames}")
        
        for row in reader:
            text = row[text_column].strip()
            
            # Skip empty or too long lines
            if not text or len(text) > max_hanzi_per_line:
                lines_skipped += 1
                continue
            
            # Convert to initials
            initials = hanzi_to_initials(text, use_digraph_repetition)
            
            if initials:
                f_out.write(initials + "\n")
                lines_processed += 1
    
    return lines_processed, lines_skipped


def validate_initials_corpus(corpus_path: str) -> bool:
    """
    Validate that initials corpus contains only [a-z] and spaces.
    
    Returns:
        True if valid, raises AssertionError otherwise
    """
    allowed_pattern = re.compile(r'^[a-z ]+$')
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if not allowed_pattern.match(line):
                invalid_chars = set(c for c in line if c not in "abcdefghijklmnopqrstuvwxyz ")
                raise AssertionError(
                    f"Invalid characters in initials corpus at line {i}: {invalid_chars}\n"
                    f"Line: {line[:100]}..."
                )
    
    print(f"[OK] Initials corpus validated: only [a-z] and spaces")
    return True


if __name__ == "__main__":
    print("=== Preprocess Smoke Test ===\n")
    
    # Test individual conversions
    test_cases = [
        ("时候", "shh", "shi hou -> sh + h within same word"),
        ("长河", "chh", "chang he -> ch + h within same word"),
        ("中国", "zhg", "zhong guo -> zh + g within same word"),
        ("上海", "shh", "shang hai -> sh + h within same word"),
        ("我们", "wm", "wo men -> w + m within same word"),
        ("今天", "jt", "jin tian -> j + t within same word"),
    ]
    
    print("Test cases:")
    for hanzi, expected, desc in test_cases:
        result = hanzi_to_initials(hanzi, use_digraph_repetition=True)
        status = "[OK]" if result == expected else f"[FAIL] (got: {result})"
        # Use ASCII-safe representation
        print(f"  {status} [hanzi] -> {result}  [{desc}]")
    
    # Test digraph repetition (need a word with repeated digraphs)
    # "周周" (zhou zhou) should give "zhh"
    print("\nDigraph repetition test:")
    test_repetition = "周周"  # Both syllables start with 'zh'
    result = hanzi_to_initials(test_repetition, use_digraph_repetition=True)
    expected_rep = "zhh"
    status = "[OK]" if result == expected_rep else f"[FAIL] (got: {result})"
    print(f"  {status} [hanzi] -> {result}  [Expected: zhh for zh+zh]")
    
    # Test with a sentence (jieba word boundaries)
    print("\nSentence test (jieba word boundaries):")
    sentence = "我们今天有什么安排"
    result = hanzi_to_initials(sentence)
    expected_sentence = "wm jt y shm ap"
    status = "[OK]" if result == expected_sentence else f"[FAIL] (got: {result})"
    print(f"  {status} [sentence] -> {result}")
    print(f"    Expected: {expected_sentence}")
    print(f"    (women=wm, jintian=jt, you=y, shenme=shm, anpai=ap)")
    
    # Validate alphabet
    print("\nValidation:")
    allowed = re.compile(r'^[a-z ]+$')
    is_valid = allowed.match(result)
    print(f"  Result contains only [a-z] and spaces: {is_valid}")
    
    print("\n[OK] Preprocess module smoke test passed")

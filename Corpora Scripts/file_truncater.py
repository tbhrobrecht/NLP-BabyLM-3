import os

def truncate_file_to_match_reference(target_file_path, reference_file_path):
    ref_size = os.path.getsize(reference_file_path)
    tgt_size = os.path.getsize(target_file_path)
    ref_size = 2*ref_size
    if tgt_size > ref_size:
        with open(target_file_path, 'r+b') as f:
            f.truncate(ref_size)
        print(f"Truncated {target_file_path} to match {reference_file_path} size ({ref_size} bytes).")
    else:
        print(f"{target_file_path} is already less than or equal to {reference_file_path} size.")

# Usage example:
# truncate_file_to_match_reference('data_saved/7mb/train.txt', 'path/to/reference.txt')

# truncate_file_to_match_reference('data_saved/7mb/train.txt', 'data_saved/7mb/initials_corpus.txt')
# truncate_file_to_match_reference('data_saved/7mb/train_pinyin.txt', 'data_saved/7mb/initials_corpus.txt')
# truncate_file_to_match_reference('data_saved/abbreviation_matches/initials_pinyin_length.txt', 'data_saved/abbreviation_matches/train_pinyin.txt')
# truncate_file_to_match_reference('data_saved/abbreviation_matches/initials_hanzi_length.txt', 'data_saved/abbreviation_matches/train.txt')
truncate_file_to_match_reference('data_saved/abbreviation_matches/initials_2x_pinyin_length.txt', 'data_saved/abbreviation_matches/train_pinyin.txt')
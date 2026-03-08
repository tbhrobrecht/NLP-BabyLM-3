import csv

# Input and output file paths
input_csv = 'data/corpus.csv'
output_txt = 'data/corpus.txt'

# Read CSV and extract text column
with open(input_csv, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # Open output file with UTF-8 encoding
    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        for row in csv_reader:
            # Write the text column followed by a newline
            txt_file.write(row['text'] + '\n')

print(f"Successfully converted {input_csv} to {output_txt}")
print(f"Text column extracted and saved with UTF-8 encoding")

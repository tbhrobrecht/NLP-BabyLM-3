import os
import json

input_dir = "blimp_pinyin_initials"
output_dir = "blimp_pinyin_initials_converted"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                data = json.loads(line)
                new_data = {
                    "sentence_good": data.get("sentence_good_initials", ""),
                    "sentence_bad": data.get("sentence_bad_initials", "")
                }
                outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
print("Conversion complete.")
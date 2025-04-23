import json
import html
import re

input_path = "pol_0616-1119_labeled/pol_062016-112019_labeled.ndjson"
output_path = "gpt2_ready_dataset.txt"

def clean_post_text(text):
    # Unescape HTML entities
    text = html.unescape(text)
    # Replace <br> with newlines
    text = re.sub(r'<br\s*/?>', '\n', text)
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Add a space after > quote markers if needed
    text = re.sub(r'(?<!\n)>(?=\S)', '> ', text)
    # Remove reply links like >>12345678
    text = re.sub(r'>>\d+', '', text)
    return text.strip()

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        try:
            thread = json.loads(line)
            for post in thread.get("posts", []):
                if "com" in post:
                    clean_text = clean_post_text(post["com"])
                    if clean_text:
                        outfile.write(clean_text + "\n<|endoftext|>\n")
        except json.JSONDecodeError:
            continue  # skip bad lines

print(f"Formatted dataset written to {output_path}")

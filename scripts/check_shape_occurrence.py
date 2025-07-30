import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from core.solution_verifier import *
import anthropic
import glob
client = anthropic.Anthropic(api_key='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')
from collections import defaultdict

def split_text(file_path, keywords):
    before_keywords = []
    after_keywords = []
    found_keyword = False
    with open(file_path, 'r', errors='ignore') as file:
        for line in file:
            # Check for the occurrence of any keyword
            if not found_keyword and any(keyword in line for keyword in keywords):
                found_keyword = True
            # Append lines to respective lists based on whether the keyword has been found
            if found_keyword:
                after_keywords.append(line)
            else:
                before_keywords.append(line)
    temp_text =(''.join(after_keywords))
    messages = []
    messages.append({"role": "user", "content": 'Given the following texts. Please extract the shape '
                                                'that the chatbot identify. Please return only one word.'})
    messages.append({"role": "user", "content": temp_text})
    response = client.messages.create(
        model='claude-3-5-sonnet-20241022',
        max_tokens=1000,
        temperature=0,
        system="You are a helpful AI assistant.",
        messages=messages,
    )
    print(temp_text)
    return response.content[0].text

def process_directory(directory, keywords):
    shape_map = defaultdict(int)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                print(f"\nProcessing file: {file_path}\n")
                shape = split_text(file_path, keywords)
                shape_map[shape] += 1
                # Write the shape information into shape.txt in the same folder as the file
                filename_without_ext = os.path.splitext(file)[0]
                shape_file_name = f"{filename_without_ext}_shape.txt"
                shape_txt_path = os.path.join(root, shape_file_name)
                # Open the file in append mode so each file's shape is appended
                with open(shape_txt_path, 'a') as shape_file:
                    shape_file.write(f"{file}: {shape}\n")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    # Save shape_map to metadata.txt in the same directory
    metadata_path = os.path.join(directory, 'metadata_dec_23.txt')
    try:
        with open(metadata_path, 'w') as metadata_file:
            for shape, count in shape_map.items():
                metadata_file.write(f"{shape}: {count}\n")
        print(f"\nMetadata saved to: {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata to {metadata_path}: {e}")

def get_args():
    parser = argparse.ArgumentParser(description="Split text into parts before and after keywords.")
    parser.add_argument("directory", type=str, help="Path to the top-level directory to search")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    keywords = [
        "SOLVE: SUCCESS",
        "SOLVE: FAIL",
        "RECOGNIZE: SUCCESS",
        "RECOGNIZE: FAIL",
        "GENERATE: SUCCESS",
        "GENERATE: FAIL",
    ]
    process_directory(args.directory, keywords)
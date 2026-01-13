import os
import json

def convert_jsonl_to_json(jsonl_file, output_folder):
    """
    Reads a .jsonl file and saves each line as an individual .json file in the output folder.
    
    :param jsonl_file: Path to the input .jsonl file.
    :param output_folder: Path to the folder where individual .json files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(jsonl_file, 'r') as file:
        for idx, line in enumerate(file):
            try:
                json_data = json.loads(line.strip())
                title = "-".join(json_data["tickers"])
                json_file_path = os.path.join(output_folder, f"{idx}_{title}.json")
                with open(json_file_path, 'w') as json_file:
                    json.dump(json_data, json_file, indent=4)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {idx}: {e}")

# Example usage
convert_jsonl_to_json("/home/ubuntu/src/timellm-data/raw_data/financial/text/sampled_content_20000_label.jsonl", "/home/ubuntu/src/timellm-data/raw_data/financial/text/news_labeled_20000")

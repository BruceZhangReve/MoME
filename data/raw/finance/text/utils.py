
import openai
import tiktoken
import json


def read_jsonl_from_local(file_path):
    """
    Reads a JSONL file from the local file system and returns a list of dictionaries.
    
    :param file_path: Path to the JSONL file.
    :return: List of dictionaries.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def save_list_of_dicts_to_jsonl(data, file_path):
    """
    Saves a list of dictionaries to a JSONL file on the local file system.
    
    :param data: List of dictionaries to save.
    :param file_path: Path to the JSONL file.
    """
    with open(file_path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')




# OpenAI API Pricing (Update if OpenAI updates their prices)
PRICING = {
    "gpt-4o": {"input": 2.5 / 1000_000, "output": 10.0 / 1000_000},  # $0.01 per 1K input tokens, $0.03 per 1K output tokens
    "gpt-4o-mini": {"input": 0.15 / 1000_000, "output": 0.6/1000_000},
    "o1": {"input": 15.0/1000_000, "output": 60.0/1000_000},
    "o1-mini": {"input": 3.0/1000_000, "output": 12.0/1000_000},
}

def count_tokens(text, model="gpt-4o"):
    """Counts tokens in a given text using OpenAI's tokenizer."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_cost(prompt, model="gpt-4o", max_response_tokens=500):
    """Estimates the cost before sending a request to OpenAI."""
    if model not in PRICING:
        raise ValueError(f"Model {model} not found in pricing dictionary. Update the PRICING dictionary.")

    input_tokens = count_tokens(prompt, model)
    output_tokens = max_response_tokens  # Estimated response length

    input_cost = input_tokens * PRICING[model]["input"]
    output_cost = output_tokens * PRICING[model]["output"]

    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }



def call_openai_api(client, prompt, model="gpt-4o", max_response_tokens=500, confirm=True):
    """
    Calls OpenAI's Chat API after estimating cost.
    
    Parameters:
        - prompt (str): User input prompt.
        - model (str): Model to use (default: gpt-4o).
        - max_response_tokens (int): Expected max response tokens (default: 500).
        - confirm (bool): Whether to ask for user confirmation before proceeding.

    Returns:
        - OpenAI API response.
    """
    # Estimate cost
    # estimate = estimate_cost(prompt, model, max_response_tokens)

    # # Display cost estimate
    # print("\nEstimated Cost Breakdown:")
    # print(f"Model: {model}")
    # print(f"Input Tokens: {estimate['input_tokens']}")
    # print(f"Output Tokens (estimated): {estimate['output_tokens']}")
    # print(f"Input Cost: ${estimate['input_cost']:.6f}")
    # print(f"Output Cost: ${estimate['output_cost']:.6f}")
    # print(f"Total Estimated Cost: ${estimate['total_cost']:.6f}")

    # # Ask for confirmation before proceeding
    # if confirm:
    #     proceed = input("Do you want to proceed with the API call? (yes/no): ").strip().lower()
    #     if proceed not in ["yes", "y"]:
    #         print("API call aborted.")
    #         return None

    # Make API call using the new OpenAI API client
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_response_tokens
    )

    # Print response
    # print("\nOpenAI API Response:")
    # print(response.choices[0].message.content)

    return response.choices[0].message.content




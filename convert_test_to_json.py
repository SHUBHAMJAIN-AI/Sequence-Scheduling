#!/usr/bin/env python3
"""
Convert GSM8K test CSV data to JSON format matching the training data structure.
"""
import pandas as pd
import json
import ast
import re

def parse_prompt_content(prompt_str):
    """Extract the math question from the prompt string."""
    try:
        # Parse the JSON-like string
        prompt_data = ast.literal_eval(prompt_str)
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            content = prompt_data[0].get("content", "")
            
            # Extract the original question (before the special instructions)
            # Look for the pattern ending with 'Let's think step by step...'
            pattern = r"(.*?)\s+Let's think step by step"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                question = match.group(1).strip()
                # Remove any trailing quotes
                question = question.strip('"').strip("'")
                return question
    except:
        pass
    return ""

def parse_extra_info(extra_info_str):
    """Extract the answer and question from extra_info."""
    try:
        extra_info = ast.literal_eval(extra_info_str)
        return extra_info.get("answer", ""), extra_info.get("question", "")
    except:
        return "", ""

def convert_csv_to_json(csv_path, output_path):
    """Convert CSV test data to JSON format for FastChat training."""
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    json_data = []
    
    for _, row in df.iterrows():
        # Extract information
        uid = row['uid']
        question = parse_prompt_content(row['prompt'])
        answer, _ = parse_extra_info(row['extra_info'])
        
        # If question extraction failed, try from extra_info
        if not question:
            _, question = parse_extra_info(row['extra_info'])
        
        if not question or not answer:
            print(f"Warning: Missing data for UID {uid}")
            continue
        
        # Create the conversation structure matching training data
        conversation_entry = {
            "id": uid,
            "conversations": [
                {
                    "from": "human",
                    "value": f"Please solve this math problem step by step. First, estimate how many tokens your response will be (give a rough number), then provide your complete solution.\n\nQuestion: {question} Let's think step by step and output the final answer after \"####\".\n\nFormat your response as:\nEstimated tokens: [your estimate]\n\nSolution: [your step-by-step solution ending with #### [final answer]]"
                },
                {
                    "from": "gpt",
                    "value": f"Estimated tokens: 50\n\nSolution: {answer}"
                }
            ]
        }
        
        json_data.append(conversation_entry)
    
    # Save to JSON
    print(f"Converting {len(json_data)} entries...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved converted data to: {output_path}")
    return len(json_data)

if __name__ == "__main__":
    csv_path = "/root/data/gsm8k/test_token_prediction.csv"
    output_path = "/root/sequence_code/Sequence-Scheduling/data/gsm8k-test-length-perception.json"
    
    num_converted = convert_csv_to_json(csv_path, output_path)
    print(f"Successfully converted {num_converted} test samples to JSON format")
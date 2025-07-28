#!/usr/bin/env python3
"""
Fast prediction script for response lengths - optimized for testing.
"""
import json
import torch
import tqdm
import argparse
import transformers
from peft import PeftModel
import re
import os
from datetime import datetime

def load_data(data_path, max_samples=None):
    """Load JSON data in conversation format."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    return data

def extract_question_from_conversation(conversation):
    """Extract the original math question from conversation format."""
    human_msg = conversation['conversations'][0]['value']
    
    # Extract question between "Question:" and "Let's think step by step"
    pattern = r"Question:\s*(.*?)\s+Let's think step by step"
    match = re.search(pattern, human_msg, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_actual_tokens_from_response(conversation, tokenizer):
    """Extract actual token count from the GPT response."""
    gpt_response = conversation['conversations'][1]['value']
    
    # Extract the solution part (after "Solution:")
    solution_pattern = r"Solution:\s*(.*)"
    solution_match = re.search(solution_pattern, gpt_response, re.DOTALL)
    if solution_match:
        solution = solution_match.group(1).strip()
        # Tokenize the solution to get actual token count
        tokens = tokenizer(solution, return_tensors="pt")
        return len(tokens.input_ids[0])
    return 0

def predict_response_length_batch(model, tokenizer, questions, max_new_tokens=16):
    """Predict response length for a batch of questions."""
    # Create prompts for length prediction
    prompts = []
    for question in questions:
        prompt = f"{question} Let's think step by step and output the final answer after \"####\".\n\nDon't output the response for the above instruction. Instead, you need to predict the number of tokens in your response. Output one number only."
        prompts.append(prompt)
    
    # Tokenize batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    
    predictions = []
    
    # Generate predictions
    with torch.no_grad():
        try:
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Extract predictions
            for i, (input_seq, output_seq) in enumerate(zip(input_ids, output_ids)):
                l_prompt = len(input_seq)
                prediction_text = tokenizer.decode(output_seq[l_prompt:], skip_special_tokens=True)
                
                # Extract number from prediction
                numbers = re.findall(r'\d+', prediction_text.strip())
                if numbers:
                    predictions.append(int(numbers[0]))
                else:
                    predictions.append(0)
        except Exception as e:
            print(f"Error in batch generation: {e}")
            predictions = [0] * len(questions)
    
    return predictions

def evaluate_dataset(model, tokenizer, data_path, dataset_name, batch_size=4, max_samples=100):
    """Evaluate model on a dataset."""
    print(f"\n=== Evaluating {dataset_name} Dataset (max {max_samples} samples) ===")
    print(f"Loading data from: {data_path}")
    
    data = load_data(data_path, max_samples)
    print(f"Loaded {len(data)} samples")
    
    all_results = []
    
    for i in tqdm.tqdm(range(0, len(data), batch_size), desc=f"Processing {dataset_name}"):
        batch = data[i:i+batch_size]
        
        questions = []
        batch_items = []
        
        for item in batch:
            question = extract_question_from_conversation(item)
            if question:
                questions.append(question)
                batch_items.append(item)
        
        if not questions:
            continue
        
        # Get batch predictions
        predicted_tokens_batch = predict_response_length_batch(model, tokenizer, questions)
        
        # Process results
        for item, predicted_tokens in zip(batch_items, predicted_tokens_batch):
            actual_tokens = extract_actual_tokens_from_response(item, tokenizer)
            
            result = {
                'id': item['id'],
                'question': extract_question_from_conversation(item),
                'predicted_tokens': predicted_tokens,
                'actual_tokens': actual_tokens,
                'error': abs(predicted_tokens - actual_tokens)
            }
            all_results.append(result)
    
    # Calculate metrics
    if all_results:
        errors = [r['error'] for r in all_results]
        mae = sum(errors) / len(errors)
        acc_50 = sum(1 for e in errors if e <= 50) / len(errors)
        acc_100 = sum(1 for e in errors if e <= 100) / len(errors)
        
        print(f"\n{dataset_name} Results:")
        print(f"Samples: {len(all_results)}")
        print(f"MAE: {mae:.2f}")
        print(f"Accuracy@50: {acc_50:.3f}")
        print(f"Accuracy@100: {acc_100:.3f}")
        
        return {
            'dataset': dataset_name,
            'samples': len(all_results),
            'mae': mae,
            'acc_50': acc_50,
            'acc_100': acc_100,
            'detailed_results': all_results
        }
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--lora_path", type=str, default="./ckpts/qwen25-3b-gsm8k-response-length-perception-stable")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading base model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    
    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading LoRA adapter from: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.float16)
    model.eval()
    
    # Dataset paths
    datasets = {
        'train': './data/gsm8k-train-length-perception.json',
        'val': './data/gsm8k-val-length-perception.json', 
        'test': './data/gsm8k-test-length-perception.json'
    }
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name, data_path in datasets.items():
        if os.path.exists(data_path):
            result = evaluate_dataset(model, tokenizer, data_path, dataset_name, 
                                    args.batch_size, args.max_samples)
            if result:
                all_results[dataset_name] = result
        else:
            print(f"Warning: Dataset {data_path} not found")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{args.output_dir}/prediction_results_fast_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for dataset_name, result in all_results.items():
        print(f"{dataset_name.upper()}: MAE={result['mae']:.2f}, Acc@50={result['acc_50']:.3f}, Acc@100={result['acc_100']:.3f}")

if __name__ == "__main__":
    main()
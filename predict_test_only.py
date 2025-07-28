#!/usr/bin/env python3
"""
Run predictions on test dataset only with fixed JSON serialization.
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
import gc
import numpy as np

def load_data(data_path):
    """Load JSON data in conversation format."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
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

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def evaluate_test_dataset(model, tokenizer, data_path, dataset_name, batch_size=8):
    """Evaluate model on test dataset."""
    print(f"\n=== Evaluating Complete {dataset_name} Dataset ===")
    print(f"Loading data from: {data_path}")
    
    data = load_data(data_path)
    total_samples = len(data)
    print(f"Loaded {total_samples} samples")
    
    all_results = []
    processed_count = 0
    
    for i in tqdm.tqdm(range(0, total_samples, batch_size), desc=f"Processing {dataset_name}"):
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
                'predicted_tokens': predicted_tokens,
                'actual_tokens': actual_tokens,
                'error': abs(predicted_tokens - actual_tokens)
            }
            all_results.append(result)
            processed_count += 1
        
        # Memory cleanup
        if processed_count % 200 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate comprehensive metrics
    if all_results:
        errors = [r['error'] for r in all_results]
        predictions = [r['predicted_tokens'] for r in all_results]
        actuals = [r['actual_tokens'] for r in all_results]
        
        # Basic metrics
        mae = sum(errors) / len(errors)
        
        # Accuracy at different thresholds
        accuracy_metrics = {}
        thresholds = [10, 25, 50, 75, 100, 150, 200]
        for threshold in thresholds:
            acc = sum(1 for e in errors if e <= threshold) / len(errors)
            accuracy_metrics[f'acc_{threshold}'] = acc
        
        # Additional statistics
        error_stats = {
            'mean': np.mean(errors),
            'median': np.median(errors),
            'std': np.std(errors),
            'min': np.min(errors),
            'max': np.max(errors),
            'q25': np.percentile(errors, 25),
            'q75': np.percentile(errors, 75),
            'q90': np.percentile(errors, 90),
            'q95': np.percentile(errors, 95)
        }
        
        prediction_stats = {
            'pred_mean': np.mean(predictions),
            'pred_std': np.std(predictions),
            'actual_mean': np.mean(actuals),
            'actual_std': np.std(actuals)
        }
        
        print(f"\n{dataset_name} Complete Results:")
        print(f"Total Samples: {len(all_results)}")
        print(f"MAE: {mae:.2f} tokens")
        print(f"Accuracy@50: {accuracy_metrics['acc_50']:.3f}")
        print(f"Accuracy@100: {accuracy_metrics['acc_100']:.3f}")
        print(f"Accuracy@150: {accuracy_metrics['acc_150']:.3f}")
        print(f"Error Range: {error_stats['min']:.0f} - {error_stats['max']:.0f} tokens")
        print(f"Error Median: {error_stats['median']:.1f} tokens")
        
        # Convert numpy types for JSON serialization
        result_data = {
            'dataset': dataset_name,
            'total_samples': len(all_results),
            'processed_samples': len(all_results),
            'mae': mae,
            'accuracy_metrics': accuracy_metrics,
            'error_statistics': error_stats,
            'prediction_statistics': prediction_stats,
            'sample_results': all_results[:50],  # Store first 50 for analysis
            'summary': {
                'mae': mae,
                'acc_50': accuracy_metrics['acc_50'],
                'acc_100': accuracy_metrics['acc_100'],
                'median_error': error_stats['median'],
                'samples': len(all_results)
            }
        }
        
        # Convert all numpy types
        result_data = convert_numpy_types(result_data)
        
        return result_data
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--lora_path", type=str, default="./ckpts/qwen25-3b-gsm8k-response-length-perception-stable")
    parser.add_argument("--batch_size", type=int, default=8)
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
    
    # Test dataset path
    data_path = './data/gsm8k-test-length-perception.json'
    
    if os.path.exists(data_path):
        print(f"\n{'='*60}")
        print(f"Starting TEST dataset evaluation...")
        print(f"{'='*60}")
        
        result = evaluate_test_dataset(model, tokenizer, data_path, 'test', args.batch_size)
        if result:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{args.output_dir}/full_results_test_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({'test': result}, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to: {results_file}")
            
            # Print summary
            print(f"\n{'='*60}")
            print("TEST DATASET FINAL RESULTS")
            print(f"{'='*60}")
            
            summary = result['summary']
            acc_metrics = result['accuracy_metrics']
            print(f"\nTEST Dataset:")
            print(f"  Samples: {summary['samples']:,}")
            print(f"  MAE: {summary['mae']:.2f} tokens")
            print(f"  Median Error: {summary['median_error']:.1f} tokens")
            print(f"  Accuracy Breakdown:")
            for threshold in [10, 25, 50, 75, 100, 150, 200]:
                acc = acc_metrics[f'acc_{threshold}']
                print(f"    Within {threshold:3d} tokens: {acc:.3f} ({acc*100:.1f}%)")
    else:
        print(f"Error: Test dataset {data_path} not found")

if __name__ == "__main__":
    main()
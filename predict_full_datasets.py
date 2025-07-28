#!/usr/bin/env python3
"""
Run predictions on complete training and test datasets for comprehensive accuracy analysis.
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

def evaluate_dataset_full(model, tokenizer, data_path, dataset_name, batch_size=4):
    """Evaluate model on complete dataset."""
    print(f"\n=== Evaluating Complete {dataset_name} Dataset ===")
    print(f"Loading data from: {data_path}")
    
    data = load_data(data_path)
    total_samples = len(data)
    print(f"Loaded {total_samples} samples")
    
    all_results = []
    processed_count = 0
    
    # Progress tracking
    checkpoint_interval = 500
    last_checkpoint = 0
    
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
                'question': extract_question_from_conversation(item)[:100] + "...",  # Truncate for storage
                'predicted_tokens': predicted_tokens,
                'actual_tokens': actual_tokens,
                'error': abs(predicted_tokens - actual_tokens)
            }
            all_results.append(result)
            processed_count += 1
        
        # Periodic checkpointing and memory cleanup
        if processed_count - last_checkpoint >= checkpoint_interval:
            print(f"Processed {processed_count}/{total_samples} samples...")
            last_checkpoint = processed_count
            
            # Memory cleanup
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
        import numpy as np
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
        
        return {
            'dataset': dataset_name,
            'total_samples': len(all_results),
            'processed_samples': len(all_results),
            'mae': mae,
            'accuracy_metrics': accuracy_metrics,
            'error_statistics': error_stats,
            'prediction_statistics': prediction_stats,
            'sample_results': all_results[:100],  # Store first 100 for analysis
            'summary': {
                'mae': mae,
                'acc_50': accuracy_metrics['acc_50'],
                'acc_100': accuracy_metrics['acc_100'],
                'median_error': error_stats['median'],
                'samples': len(all_results)
            }
        }
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--lora_path", type=str, default="./ckpts/qwen25-3b-gsm8k-response-length-perception-stable")
    parser.add_argument("--batch_size", type=int, default=6)
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
    
    # Dataset paths for full evaluation
    datasets = {
        'train': './data/gsm8k-train-length-perception.json',
        'test': './data/gsm8k-test-length-perception.json'
    }
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name, data_path in datasets.items():
        if os.path.exists(data_path):
            print(f"\n{'='*60}")
            print(f"Starting {dataset_name.upper()} dataset evaluation...")
            print(f"{'='*60}")
            
            result = evaluate_dataset_full(model, tokenizer, data_path, dataset_name, args.batch_size)
            if result:
                all_results[dataset_name] = result
                
                # Save intermediate results
                intermediate_file = f"{args.output_dir}/full_results_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump({dataset_name: result}, f, indent=2, ensure_ascii=False)
                print(f"Intermediate results saved to: {intermediate_file}")
        else:
            print(f"Warning: Dataset {data_path} not found")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{args.output_dir}/full_prediction_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("FINAL COMPREHENSIVE RESULTS")
    print(f"{'='*60}")
    print(f"Results saved to: {results_file}")
    
    # Print comprehensive summary
    print("\n=== COMPLETE DATASET ACCURACY ANALYSIS ===")
    for dataset_name, result in all_results.items():
        summary = result['summary']
        acc_metrics = result['accuracy_metrics']
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"  Samples: {summary['samples']:,}")
        print(f"  MAE: {summary['mae']:.2f} tokens")
        print(f"  Median Error: {summary['median_error']:.1f} tokens")
        print(f"  Accuracy Breakdown:")
        for threshold in [10, 25, 50, 75, 100, 150, 200]:
            acc = acc_metrics[f'acc_{threshold}']
            print(f"    Within {threshold:3d} tokens: {acc:.3f} ({acc*100:.1f}%)")
    
    # Overall comparison
    if len(all_results) == 2:
        train_mae = all_results['train']['mae']
        test_mae = all_results['test']['mae']
        generalization_gap = abs(train_mae - test_mae)
        print(f"\nGeneralization Analysis:")
        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Test MAE:  {test_mae:.2f}")
        print(f"  Gap:       {generalization_gap:.2f} tokens")
        
        if generalization_gap < 5:
            print("  Status: Excellent generalization")
        elif generalization_gap < 10:
            print("  Status: Good generalization")
        else:
            print("  Status: Potential overfitting detected")

if __name__ == "__main__":
    main()
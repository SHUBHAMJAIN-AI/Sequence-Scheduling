#!/usr/bin/env python3
"""
Prediction Cost Measurement Tool for Sequential-Scheduling

This script measures prediction costs including:
- GPU time (CUDA time)
- Wall-clock prediction time
- GPU memory usage
- Throughput metrics
- Token generation costs

Usage:
    python measure_prediction_costs.py --model ./ckpts/qwen25-3b-gsm8k-response-length-perception-stable --data-path data/gsm8k-test-length-perception.json
"""

import argparse
import json
import time
import psutil
import torch
import torch.cuda
import numpy as np
from typing import Dict, List, Tuple
import logging
import os
from datetime import datetime

# Import existing modules
from src.utils import timeit, jdump, EvalDataset, set_seed
from src.generate import Creator


class PredictionCostMeasurer:
    def __init__(self, model_path: str, lora_path: str = None, debug: bool = False):
        """Initialize the cost measurement tool."""
        self.model_path = model_path
        self.lora_path = lora_path
        self.debug = debug
        
        # Initialize model
        print(f"Loading model from: {model_path}")
        if lora_path:
            print(f"Loading LoRA adapter from: {lora_path}")
        
        self.model = Creator(model_path, debug=debug, lora_path=lora_path)
        
        # Fix tokenizer padding issue
        if self.model.tokenizer.pad_token is None:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
            print("Fixed tokenizer padding token")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def measure_gpu_memory(self) -> Dict[str, float]:
        """Measure GPU memory usage."""
        if not torch.cuda.is_available():
            return {"gpu_memory_allocated": 0.0, "gpu_memory_reserved": 0.0}
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        
        return {
            "gpu_memory_allocated_gb": memory_allocated,
            "gpu_memory_reserved_gb": memory_reserved
        }
    
    def measure_cpu_memory(self) -> Dict[str, float]:
        """Measure CPU memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "cpu_memory_rss_gb": memory_info.rss / 1024**3,  # GB
            "cpu_memory_vms_gb": memory_info.vms / 1024**3   # GB
        }
    
    def run_inference_only(self, inputs: List[str], batch_size: int = 128) -> Dict:
        """Run inference-only prediction (no text generation) for batch processing cost measurement."""
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Measure initial memory
        initial_gpu_memory = self.measure_gpu_memory()
        initial_cpu_memory = self.measure_cpu_memory()
        
        # Prepare tokenized inputs
        tokenizer, model, device = self.model.tokenizer, self.model.model, self.model.device
        
        # Process inputs in batches
        total_samples = len(inputs)
        batch_timings = []
        total_tokens_processed = 0
        
        print(f"Processing {total_samples} samples with batch size {batch_size}...")
        
        for i in range(0, total_samples, batch_size):
            batch_inputs = inputs[i:i + batch_size]
            actual_batch_size = len(batch_inputs)
            
            # Start batch timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_start = time.time()
            cuda_batch_start = timeit() if torch.cuda.is_available() else batch_start
            
            # Tokenize batch
            try:
                tokenized = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)
                
                # Forward pass only (no generation)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Count tokens processed
                tokens_in_batch = input_ids.numel()  # Total tokens in batch
                total_tokens_processed += tokens_in_batch
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue
            
            # End batch timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_end = time.time()
            cuda_batch_end = timeit() if torch.cuda.is_available() else batch_end
            
            # Measure memory after this batch
            if torch.cuda.is_available():
                memory_after_batch = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved_after_batch = torch.cuda.memory_reserved() / 1024**3  # GB
            else:
                memory_after_batch = 0.0
                memory_reserved_after_batch = 0.0
            
            # Record batch timing and memory
            batch_wall_time = batch_end - batch_start
            batch_cuda_time = cuda_batch_end - cuda_batch_start if torch.cuda.is_available() else batch_wall_time
            
            batch_timings.append({
                "batch": i // batch_size,
                "samples_in_batch": actual_batch_size,
                "tokens_in_batch": tokens_in_batch,
                "wall_time_s": batch_wall_time,
                "cuda_time_s": batch_cuda_time,
                "samples_per_second": actual_batch_size / batch_cuda_time if batch_cuda_time > 0 else 0,
                "tokens_per_second": tokens_in_batch / batch_cuda_time if batch_cuda_time > 0 else 0,
                "memory_after_batch_gb": memory_after_batch,
                "memory_reserved_after_batch_gb": memory_reserved_after_batch
            })
            
            # Progress report
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}, "
                      f"time: {batch_cuda_time:.4f}s, throughput: {actual_batch_size/batch_cuda_time:.1f} samples/s, "
                      f"memory: {memory_after_batch:.1f}GB")
        
        # Measure final memory
        final_gpu_memory = self.measure_gpu_memory()
        final_cpu_memory = self.measure_cpu_memory()
        
        # Calculate aggregate statistics
        total_wall_time = sum(b["wall_time_s"] for b in batch_timings)
        total_cuda_time = sum(b["cuda_time_s"] for b in batch_timings)
        
        # Remove any empty batches
        valid_batches = [b for b in batch_timings if b["samples_in_batch"] > 0]
        
        if not valid_batches:
            return {"error": "No valid batches processed"}
        
        # Calculate per-step statistics
        avg_cuda_time_per_step = total_cuda_time / len(valid_batches)
        avg_samples_per_step = sum(b["samples_in_batch"] for b in valid_batches) / len(valid_batches)
        avg_tokens_per_step = sum(b["tokens_in_batch"] for b in valid_batches) / len(valid_batches)
        
        # Calculate memory statistics
        if valid_batches and "memory_after_batch_gb" in valid_batches[0]:
            memory_per_step = [b["memory_after_batch_gb"] for b in valid_batches]
            memory_reserved_per_step = [b["memory_reserved_after_batch_gb"] for b in valid_batches]
            
            avg_memory_per_step = sum(memory_per_step) / len(memory_per_step)
            min_memory_per_step = min(memory_per_step)
            max_memory_per_step = max(memory_per_step)
            avg_memory_reserved_per_step = sum(memory_reserved_per_step) / len(memory_reserved_per_step)
            
            memory_stats = {
                "avg_memory_per_step_gb": avg_memory_per_step,
                "min_memory_per_step_gb": min_memory_per_step,
                "max_memory_per_step_gb": max_memory_per_step,
                "avg_memory_reserved_per_step_gb": avg_memory_reserved_per_step,
                "memory_per_sample_gb": avg_memory_per_step / avg_samples_per_step,
                "memory_measurements_available": True
            }
        else:
            memory_stats = {"memory_measurements_available": False}
        
        return {
            "timing": {
                "total_wall_time_s": total_wall_time,
                "total_cuda_time_s": total_cuda_time,
                "total_batches_processed": len(valid_batches),
                "avg_cuda_time_per_step": avg_cuda_time_per_step,
                "batch_timings": batch_timings
            },
            "memory": {
                "initial_gpu_memory_gb": initial_gpu_memory["gpu_memory_allocated_gb"],
                "final_gpu_memory_gb": final_gpu_memory["gpu_memory_allocated_gb"],
                "gpu_memory_peak_gb": final_gpu_memory["gpu_memory_reserved_gb"],
                "gpu_memory_used_gb": final_gpu_memory["gpu_memory_allocated_gb"] - initial_gpu_memory["gpu_memory_allocated_gb"],
                "initial_cpu_memory_gb": initial_cpu_memory["cpu_memory_rss_gb"],
                "final_cpu_memory_gb": final_cpu_memory["cpu_memory_rss_gb"],
                "cpu_memory_used_gb": final_cpu_memory["cpu_memory_rss_gb"] - initial_cpu_memory["cpu_memory_rss_gb"]
            },
            "processing": {
                "total_samples": total_samples,
                "total_tokens_processed": total_tokens_processed,
                "avg_samples_per_step": avg_samples_per_step,
                "avg_tokens_per_step": avg_tokens_per_step,
                "configured_batch_size": batch_size
            },
            "throughput": {
                "samples_per_second": total_samples / total_cuda_time if total_cuda_time > 0 else 0,
                "tokens_per_second": total_tokens_processed / total_cuda_time if total_cuda_time > 0 else 0,
                "steps_per_second": len(valid_batches) / total_cuda_time if total_cuda_time > 0 else 0
            },
            "costs": {
                "total_gpu_seconds": total_cuda_time,
                "gpu_seconds_per_step": avg_cuda_time_per_step,
                "gpu_seconds_per_sample": total_cuda_time / total_samples if total_samples > 0 else 0,
                "gpu_memory_gb_seconds": final_gpu_memory["gpu_memory_allocated_gb"] * total_cuda_time,
            },
            "step_memory_analysis": memory_stats
        }

    def run_single_prediction_with_step_timing(self, inputs: List[str], ids: List[str], 
                            strategy: str = "seqsch", max_length: int = 512,
                            temperature: float = 0.0, batch_size: int = 1) -> Dict:
        """Run prediction with detailed per-step timing."""
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Measure initial memory
        initial_gpu_memory = self.measure_gpu_memory()
        initial_cpu_memory = self.measure_cpu_memory()
        
        # Start timing
        start_time = time.time()
        cuda_start_time = timeit() if torch.cuda.is_available() else start_time
        
        # Run prediction with step-by-step timing
        try:
            if strategy == "vanilla":
                # Modified to capture step timing for vanilla strategy
                results, step_timings = self.generate_with_step_timing(
                    inputs, max_length=max_length, temperature=temperature
                )
            else:
                # For other strategies, fall back to regular timing
                results = self.model(
                    inputs,
                    strategy="group",
                    perception_strategy=strategy,
                    vbs=False,
                    fcr=False,
                    temperature=temperature,
                    max_length=max_length,
                    batch_size=batch_size,
                    ids=ids
                )
                step_timings = []
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}
        
        # End timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        cuda_end_time = timeit() if torch.cuda.is_available() else time.time()
        end_time = time.time()
        
        # Measure final memory
        final_gpu_memory = self.measure_gpu_memory()
        final_cpu_memory = self.measure_cpu_memory()
        
        # Calculate costs
        wall_clock_time = end_time - start_time
        cuda_time = cuda_end_time - cuda_start_time if torch.cuda.is_available() else wall_clock_time
        
        # Process results
        total_input_tokens = sum(item.get("num_input_tokens", 0) for item in results)
        total_output_tokens = sum(item.get("num_output_tokens", 0) for item in results)
        total_tokens = total_input_tokens + total_output_tokens
        
        finished_samples = sum(1 for item in results if item.get("is_finished", False))
        
        return {
            "timing": {
                "wall_clock_time_s": wall_clock_time,
                "cuda_time_s": cuda_time,
                "gpu_time_factor": cuda_time / wall_clock_time if wall_clock_time > 0 else 1.0,
                "step_timings": step_timings  # Detailed per-step timing
            },
            "memory": {
                "initial_gpu_memory_gb": initial_gpu_memory["gpu_memory_allocated_gb"],
                "final_gpu_memory_gb": final_gpu_memory["gpu_memory_allocated_gb"],
                "gpu_memory_peak_gb": final_gpu_memory["gpu_memory_reserved_gb"],
                "gpu_memory_used_gb": final_gpu_memory["gpu_memory_allocated_gb"] - initial_gpu_memory["gpu_memory_allocated_gb"],
                "initial_cpu_memory_gb": initial_cpu_memory["cpu_memory_rss_gb"],
                "final_cpu_memory_gb": final_cpu_memory["cpu_memory_rss_gb"],
                "cpu_memory_used_gb": final_cpu_memory["cpu_memory_rss_gb"] - initial_cpu_memory["cpu_memory_rss_gb"]
            },
            "tokens": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens
            },
            "throughput": {
                "samples_per_second": len(inputs) / wall_clock_time if wall_clock_time > 0 else 0,
                "tokens_per_second": total_tokens / wall_clock_time if wall_clock_time > 0 else 0,
                "cuda_tokens_per_second": total_tokens / cuda_time if cuda_time > 0 else 0
            },
            "efficiency": {
                "finished_samples": finished_samples,
                "completion_rate": finished_samples / len(inputs) if len(inputs) > 0 else 0,
                "avg_tokens_per_sample": total_tokens / len(inputs) if len(inputs) > 0 else 0
            },
            "costs": {
                "gpu_seconds": cuda_time,
                "gpu_memory_gb_seconds": final_gpu_memory["gpu_memory_allocated_gb"] * cuda_time,
                "cost_per_token": cuda_time / total_tokens if total_tokens > 0 else 0,
                "cost_per_sample": cuda_time / len(inputs) if len(inputs) > 0 else 0,
                "avg_gpu_time_per_step": sum(s["cuda_time_s"] for s in step_timings) / len(step_timings) if step_timings else 0,
                "total_generation_steps": len(step_timings)
            },
            "raw_results": results
        }

    def generate_with_step_timing(self, prompt, max_length=256, temperature=0.0):
        """Modified generate function that captures timing for each step."""
        tokenizer, model, device = self.model.tokenizer, self.model.model, self.model.device
        
        # Preparation
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        l_input_ids = len(input_ids[0])
        output_ids = input_ids
        attention_mask = inputs.attention_mask.to(device)
        ending = [-1] * len(prompt)
        
        step_timings = []
        
        print(f"Starting generation with {max_length} max steps...")
        
        for i in range(max_length):
            # Start step timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.time()
            cuda_step_start = timeit() if torch.cuda.is_available() else step_start
            
            # Generation step
            if i == 0:
                out = model(input_ids, use_cache=True, attention_mask=attention_mask)
            else:
                out = model(
                    input_ids=token,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
            
            # Sample
            last_token_logits = out.logits[:, -1]
            token = self.model.sample(last_token_logits, temperature)
            output_ids = torch.cat((output_ids, token), dim=1)
            
            # Update attention mask and kv cache
            past_key_values = out.past_key_values
            attn_dtype = attention_mask.dtype
            extend_mask = torch.ones(len(token), 1, dtype=attn_dtype).to(device)
            attention_mask = torch.cat((attention_mask, extend_mask), dim=1)
            
            # End step timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_end = time.time()
            cuda_step_end = timeit() if torch.cuda.is_available() else step_end
            
            # Record step timing
            step_wall_time = step_end - step_start
            step_cuda_time = cuda_step_end - cuda_step_start if torch.cuda.is_available() else step_wall_time
            
            step_timings.append({
                "step": i,
                "wall_time_s": step_wall_time,
                "cuda_time_s": step_cuda_time,
                "tokens_generated": len(token),
                "total_tokens_so_far": output_ids.shape[1]
            })
            
            # Ending detection
            num_ended = 0
            for j in range(len(prompt)):
                if ending[j] == -1 and token[j] == tokenizer.eos_token_id:
                    ending[j] = i
                if ending[j] != -1:
                    num_ended += 1
            
            if num_ended == len(prompt):
                print(f"All sequences finished at step {i}")
                break
            
            # Progress report every 50 steps
            if (i + 1) % 50 == 0:
                print(f"Completed step {i+1}/{max_length}, avg step time: {step_cuda_time:.4f}s")
        
        # Collect results (same as original)
        results = []
        for i in range(len(output_ids)):
            if ending[i] != -1:
                output_ = output_ids[i][: l_input_ids + ending[i]]
                is_finished = True
            else:
                output_ = output_ids[i]
                is_finished = False
            sentence = tokenizer.decode(output_, skip_special_tokens=True)
            output = sentence[len(prompt[i]) :]
            
            num_input_tokens = len(input_ids[i])
            num_output_tokens = len(tokenizer(output).input_ids)
            num_total_tokens = num_input_tokens + num_output_tokens
            length = output_ids[i].shape[0] - l_input_ids + 1
            
            result = dict(
                input=prompt[i],
                output=output,
                sentence=sentence,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                num_total_tokens=num_total_tokens,
                is_finished=is_finished,
                length=length,
            )
            results.append(result)
        
        return results, step_timings

    def run_single_prediction(self, inputs: List[str], ids: List[str], 
                            strategy: str = "seqsch", max_length: int = 512,
                            temperature: float = 0.0, batch_size: int = 1) -> Dict:
        """Run a single prediction and measure costs."""
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Measure initial memory
        initial_gpu_memory = self.measure_gpu_memory()
        initial_cpu_memory = self.measure_cpu_memory()
        
        # Start timing
        start_time = time.time()
        cuda_start_time = timeit() if torch.cuda.is_available() else start_time
        
        # Run prediction - use the same approach as the original benchmark
        try:
            if strategy == "vanilla":
                # For vanilla strategy, use batch generation directly
                results = self.model(
                    inputs,
                    strategy="batch",
                    max_length=max_length,
                    temperature=temperature
                )
            else:
                # For seqsch and other strategies, need group strategy
                results = self.model(
                    inputs,
                    strategy="group",
                    perception_strategy=strategy,
                    vbs=False,  # Variable batch size
                    fcr=False,  # First-come-first-served  
                    temperature=temperature,
                    max_length=max_length,
                    batch_size=batch_size,
                    ids=ids
                )
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}
        
        # End timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        cuda_end_time = timeit() if torch.cuda.is_available() else time.time()
        end_time = time.time()
        
        # Measure final memory
        final_gpu_memory = self.measure_gpu_memory()
        final_cpu_memory = self.measure_cpu_memory()
        
        # Calculate costs
        wall_clock_time = end_time - start_time
        cuda_time = cuda_end_time - cuda_start_time if torch.cuda.is_available() else wall_clock_time
        
        # Process results
        total_input_tokens = sum(item.get("num_input_tokens", 0) for item in results)
        total_output_tokens = sum(item.get("num_output_tokens", 0) for item in results)
        total_tokens = total_input_tokens + total_output_tokens
        
        finished_samples = sum(1 for item in results if item.get("is_finished", False))
        
        return {
            "timing": {
                "wall_clock_time_s": wall_clock_time,
                "cuda_time_s": cuda_time,
                "gpu_time_factor": cuda_time / wall_clock_time if wall_clock_time > 0 else 1.0
            },
            "memory": {
                "initial_gpu_memory_gb": initial_gpu_memory["gpu_memory_allocated_gb"],
                "final_gpu_memory_gb": final_gpu_memory["gpu_memory_allocated_gb"],
                "gpu_memory_peak_gb": final_gpu_memory["gpu_memory_reserved_gb"],
                "gpu_memory_used_gb": final_gpu_memory["gpu_memory_allocated_gb"] - initial_gpu_memory["gpu_memory_allocated_gb"],
                "initial_cpu_memory_gb": initial_cpu_memory["cpu_memory_rss_gb"],
                "final_cpu_memory_gb": final_cpu_memory["cpu_memory_rss_gb"],
                "cpu_memory_used_gb": final_cpu_memory["cpu_memory_rss_gb"] - initial_cpu_memory["cpu_memory_rss_gb"]
            },
            "tokens": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens
            },
            "throughput": {
                "samples_per_second": len(inputs) / wall_clock_time if wall_clock_time > 0 else 0,
                "tokens_per_second": total_tokens / wall_clock_time if wall_clock_time > 0 else 0,
                "cuda_tokens_per_second": total_tokens / cuda_time if cuda_time > 0 else 0
            },
            "efficiency": {
                "finished_samples": finished_samples,
                "completion_rate": finished_samples / len(inputs) if len(inputs) > 0 else 0,
                "avg_tokens_per_sample": total_tokens / len(inputs) if len(inputs) > 0 else 0
            },
            "costs": {
                "gpu_seconds": cuda_time,
                "gpu_memory_gb_seconds": final_gpu_memory["gpu_memory_allocated_gb"] * cuda_time,
                "cost_per_token": cuda_time / total_tokens if total_tokens > 0 else 0,
                "cost_per_sample": cuda_time / len(inputs) if len(inputs) > 0 else 0
            },
            "raw_results": results
        }
    
    def benchmark_dataset(self, data_path: str, num_samples: int = None, 
                         strategies: List[str] = ["seqsch", "vanilla"],
                         batch_sizes: List[int] = [1, 4, 8],
                         max_lengths: List[int] = [256, 512],
                         seed: int = 42) -> Dict:
        """Benchmark prediction costs across different configurations."""
        
        # Load dataset
        dataset = EvalDataset(data_path)
        if num_samples:
            dataset.sample(num_samples, seed=seed)
        
        print(f"Benchmarking with {len(dataset)} samples")
        
        all_results = {}
        
        for strategy in strategies:
            for batch_size in batch_sizes:
                for max_length in max_lengths:
                    config_name = f"{strategy}_bs{batch_size}_ml{max_length}"
                    print(f"\nRunning configuration: {config_name}")
                    
                    # Prepare data batches
                    batch_results = []
                    
                    for i in range(0, len(dataset), batch_size):
                        batch_data = dataset[i:i + batch_size]
                        
                        if isinstance(batch_data["input"], str):
                            inputs = [batch_data["input"]]
                            ids = [batch_data["id"]]
                        else:
                            inputs = batch_data["input"]
                            ids = batch_data["id"]
                        
                        # Run prediction and measure costs with step timing
                        result = self.run_single_prediction_with_step_timing(
                            inputs=inputs,
                            ids=ids,
                            strategy=strategy,
                            max_length=max_length,
                            batch_size=batch_size
                        )
                        
                        if "error" not in result:
                            batch_results.append(result)
                        else:
                            print(f"Batch {i//batch_size} failed: {result['error']}")
                    
                    # Aggregate results
                    if batch_results:
                        all_results[config_name] = self.aggregate_results(batch_results)
                        print(f"Completed {config_name}: {len(batch_results)} batches processed")
                    else:
                        print(f"No successful batches for {config_name}")
        
        return all_results
    
    def benchmark_inference_only(self, data_path: str, num_samples: int = None, 
                                batch_size: int = 128, seed: int = 42) -> Dict:
        """Benchmark inference-only prediction costs for batch processing."""
        
        # Load dataset
        dataset = EvalDataset(data_path)
        if num_samples:
            dataset.sample(num_samples, seed=seed)
        
        print(f"Running inference-only benchmark with {len(dataset)} samples, batch size {batch_size}")
        
        # Collect all inputs
        inputs = []
        for i in range(len(dataset)):
            sample = dataset[i]
            inputs.append(sample["input"])
        
        # Run inference-only measurement
        result = self.run_inference_only(inputs, batch_size=batch_size)
        
        if "error" in result:
            print(f"Inference failed: {result['error']}")
            return {}
        
        return {"inference_batch_{}".format(batch_size): result}
    
    def aggregate_results(self, batch_results: List[Dict]) -> Dict:
        """Aggregate results from multiple batches."""
        if not batch_results:
            return {}
        
        # Sum values
        total_wall_time = sum(r["timing"]["wall_clock_time_s"] for r in batch_results)
        total_cuda_time = sum(r["timing"]["cuda_time_s"] for r in batch_results)
        total_tokens = sum(r["tokens"]["total_tokens"] for r in batch_results)
        total_samples = sum(len(r["raw_results"]) for r in batch_results)
        total_finished = sum(r["efficiency"]["finished_samples"] for r in batch_results)
        
        # Average values
        avg_gpu_memory = np.mean([r["memory"]["gpu_memory_used_gb"] for r in batch_results])
        avg_cpu_memory = np.mean([r["memory"]["cpu_memory_used_gb"] for r in batch_results])
        
        # Aggregate step timing data
        all_step_timings = []
        total_generation_steps = 0
        avg_step_times = []
        
        for result in batch_results:
            step_timings = result["timing"].get("step_timings", [])
            all_step_timings.extend(step_timings)
            total_generation_steps += len(step_timings)
            if step_timings:
                avg_step_times.append(np.mean([s["cuda_time_s"] for s in step_timings]))
        
        # Calculate step timing statistics
        if all_step_timings:
            step_cuda_times = [s["cuda_time_s"] for s in all_step_timings]
            step_wall_times = [s["wall_time_s"] for s in all_step_timings]
            
            step_timing_stats = {
                "total_generation_steps": total_generation_steps,
                "avg_gpu_time_per_step": np.mean(step_cuda_times),
                "min_gpu_time_per_step": np.min(step_cuda_times),
                "max_gpu_time_per_step": np.max(step_cuda_times),
                "std_gpu_time_per_step": np.std(step_cuda_times),
                "avg_wall_time_per_step": np.mean(step_wall_times),
                "min_wall_time_per_step": np.min(step_wall_times),
                "max_wall_time_per_step": np.max(step_wall_times),
                "std_wall_time_per_step": np.std(step_wall_times),
                "first_step_avg_time": np.mean([s["cuda_time_s"] for s in all_step_timings if s["step"] == 0]) if any(s["step"] == 0 for s in all_step_timings) else 0,
                "subsequent_steps_avg_time": np.mean([s["cuda_time_s"] for s in all_step_timings if s["step"] > 0]) if any(s["step"] > 0 for s in all_step_timings) else 0
            }
        else:
            step_timing_stats = {}
        
        return {
            "summary": {
                "total_batches": len(batch_results),
                "total_samples": total_samples,
                "total_tokens": total_tokens,
                "total_wall_time_s": total_wall_time,
                "total_cuda_time_s": total_cuda_time,
                "total_finished_samples": total_finished
            },
            "averages": {
                "avg_wall_time_per_batch_s": total_wall_time / len(batch_results),
                "avg_cuda_time_per_batch_s": total_cuda_time / len(batch_results),
                "avg_tokens_per_batch": total_tokens / len(batch_results),
                "avg_gpu_memory_used_gb": avg_gpu_memory,
                "avg_cpu_memory_used_gb": avg_cpu_memory
            },
            "throughput": {
                "samples_per_second": total_samples / total_wall_time if total_wall_time > 0 else 0,
                "tokens_per_second": total_tokens / total_wall_time if total_wall_time > 0 else 0,
                "cuda_tokens_per_second": total_tokens / total_cuda_time if total_cuda_time > 0 else 0
            },
            "costs": {
                "total_gpu_seconds": total_cuda_time,
                "total_gpu_memory_gb_seconds": avg_gpu_memory * total_cuda_time,
                "cost_per_token": total_cuda_time / total_tokens if total_tokens > 0 else 0,
                "cost_per_sample": total_cuda_time / total_samples if total_samples > 0 else 0
            },
            "efficiency": {
                "completion_rate": total_finished / total_samples if total_samples > 0 else 0,
                "avg_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0
            },
            "step_timing_analysis": step_timing_stats,
            "batch_details": batch_results
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Measure prediction costs for sequential-scheduling")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Data configuration
    parser.add_argument("--data-path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Benchmark configuration
    parser.add_argument("--strategies", nargs="+", default=["seqsch", "vanilla"], 
                       choices=["seqsch", "vanilla", "gt", "po"], help="Prediction strategies to test")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8], 
                       help="Batch sizes to test")
    parser.add_argument("--max-lengths", nargs="+", type=int, default=[256, 512], 
                       help="Max lengths to test")
    parser.add_argument("--inference-only", action="store_true", 
                       help="Run inference-only mode (no text generation)")
    parser.add_argument("--inference-batch-size", type=int, default=128,
                       help="Batch size for inference-only mode")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./cost_analysis", 
                       help="Output directory for results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize cost measurer
    print("Initializing prediction cost measurement...")
    measurer = PredictionCostMeasurer(
        model_path=args.model,
        lora_path=args.lora_path,
        debug=args.debug
    )
    
    # Run benchmark
    print("Starting benchmark...")
    if args.inference_only:
        # Run inference-only benchmark
        results = measurer.benchmark_inference_only(
            data_path=args.data_path,
            num_samples=args.num_samples,
            batch_size=args.inference_batch_size,
            seed=args.seed
        )
    else:
        # Run regular text generation benchmark
        results = measurer.benchmark_dataset(
            data_path=args.data_path,
            num_samples=args.num_samples,
            strategies=args.strategies,
            batch_sizes=args.batch_sizes,
            max_lengths=args.max_lengths,
            seed=args.seed
        )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"prediction_costs_{timestamp}.json")
    jdump(results, output_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PREDICTION COST ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for config_name, config_results in results.items():
        if not config_results:
            continue
        
        print(f"\nConfiguration: {config_name}")
        
        # Handle inference-only results
        if "inference_batch_" in config_name:
            timing = config_results["timing"]
            costs = config_results["costs"]
            throughput = config_results["throughput"]
            processing = config_results["processing"]
            
            print(f"  Total samples: {processing['total_samples']}")
            print(f"  Batch size: {processing['configured_batch_size']}")
            print(f"  Total batches processed: {timing['total_batches_processed']}")
            print(f"  Total GPU time: {costs['total_gpu_seconds']:.2f} seconds")
            print(f"\n  PER-STEP COSTS (BATCH PROCESSING):")
            print(f"    GPU time per step: {costs['gpu_seconds_per_step']:.4f} seconds")
            print(f"    Samples per step: {processing['avg_samples_per_step']:.1f}")
            print(f"    GPU time per sample: {costs['gpu_seconds_per_sample']:.6f} seconds")
            print(f"    Throughput: {throughput['samples_per_second']:.1f} samples/s")
            print(f"    Steps per second: {throughput['steps_per_second']:.2f} steps/s")
            print(f"    GPU Memory usage: {config_results['memory']['gpu_memory_peak_gb']:.2f} GB")
            
            # Show per-step memory analysis if available
            step_memory = config_results.get('step_memory_analysis', {})
            if step_memory.get('memory_measurements_available', False):
                print(f"\n  PER-STEP MEMORY ANALYSIS:")
                print(f"    Average memory per step: {step_memory['avg_memory_per_step_gb']:.2f} GB")
                print(f"    Min memory per step: {step_memory['min_memory_per_step_gb']:.2f} GB")
                print(f"    Max memory per step: {step_memory['max_memory_per_step_gb']:.2f} GB")
                print(f"    Memory per sample: {step_memory['memory_per_sample_gb']:.4f} GB ({step_memory['memory_per_sample_gb']*1024:.1f} MB)")
                print(f"    Reserved memory per step: {step_memory['avg_memory_reserved_per_step_gb']:.2f} GB")
            
            # Estimate full dataset cost
            if processing['total_samples'] > 0:
                full_dataset_estimate = costs['gpu_seconds_per_sample'] * 10000  # Assume 10k samples
                print(f"\n  FULL DATASET COST ESTIMATE (10,000 samples):")
                print(f"    Total GPU time: {full_dataset_estimate:.1f} seconds ({full_dataset_estimate/3600:.2f} hours)")
                print(f"    Total steps needed: {10000 / processing['avg_samples_per_step']:.0f}")
        
        else:
            # Handle regular generation results
            summary = config_results["summary"]
            costs = config_results["costs"]
            throughput = config_results["throughput"]
            step_timing = config_results.get("step_timing_analysis", {})
            
            print(f"  Total samples: {summary['total_samples']}")
            print(f"  Total GPU time: {costs['total_gpu_seconds']:.2f} seconds")
            print(f"  GPU cost per sample: {costs['cost_per_sample']:.4f} s/sample")
            print(f"  GPU cost per token: {costs['cost_per_token']:.6f} s/token")
            print(f"  Throughput: {throughput['samples_per_second']:.2f} samples/s")
            print(f"  Token throughput: {throughput['tokens_per_second']:.2f} tokens/s")
            print(f"  Completion rate: {config_results['efficiency']['completion_rate']:.2%}")
            
            # Step timing information
            if step_timing:
                print(f"\n  STEP TIMING ANALYSIS:")
                print(f"    Total generation steps: {step_timing['total_generation_steps']}")
                print(f"    Average GPU time per step: {step_timing['avg_gpu_time_per_step']:.4f} seconds")
                print(f"    Min GPU time per step: {step_timing['min_gpu_time_per_step']:.4f} seconds")
                print(f"    Max GPU time per step: {step_timing['max_gpu_time_per_step']:.4f} seconds")
                print(f"    Standard deviation: {step_timing['std_gpu_time_per_step']:.4f} seconds")
                print(f"    First step avg time: {step_timing['first_step_avg_time']:.4f} seconds")
                print(f"    Subsequent steps avg time: {step_timing['subsequent_steps_avg_time']:.4f} seconds")
    
    print(f"\nDetailed results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
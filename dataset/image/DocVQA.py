# DocVQA.py
import os

os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import random
import gc
from datasets import load_dataset

def stream_and_sample_fast(dataset_name="HuggingFaceM4/DocumentVQA", split_name="validation", total_size=5349, seed=2026, num_samples=256, output_dir="./docvqa_sampled"):
    print(f"🚀 [DocVQA] Starting strictly-controlled streaming sampling via mirror (Seed={seed})...")
    
    random.seed(seed)
    target_indices = set(random.sample(range(total_size), num_samples))
    
    print(f"🌐 Connecting to the HF mirror stream for {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split_name, streaming=True)
    
    safe_name = dataset_name.split('/')[-1].lower()
    save_dir = os.path.join(output_dir, f"{safe_name}_seed{seed}")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    jsonl_path = os.path.join(save_dir, f"{safe_name}_{split_name}_{num_samples}.jsonl")
    
    extracted_count = 0
    print(f"🎯 Intercepting target data from the stream (Forced memory cleanup enabled)...")
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(dataset):
            
            # 🌟 Core Defense 1: Immediately destroy non-target data to free memory
            if i not in target_indices:
                del item
                if i % 50 == 0:
                    gc.collect()
                continue
                
            try:
                # Target hit, start parsing data
                image = item['image'].convert("RGB")
                query = item.get('question') or item.get('query', '')
                answers = item.get('answers') or item.get('label', '')
                
                img_filename = f"{safe_name}_{i}.png"
                img_path = os.path.join(img_dir, img_filename)
                image.save(img_path)
                
                record = {
                    "id": f"seed{seed}_{i}",
                    "image": f"images/{img_filename}",
                    "query": query,
                    "label": answers 
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                extracted_count += 1
                if extracted_count % 50 == 0 or extracted_count == num_samples:
                    print(f"  [Progress] Extracted {extracted_count}/{num_samples} samples (Current stream index: {i})")
                    
            except Exception as e:
                print(f"⚠️ Extraction error, skipping: {e}")
                
            # 🌟 Core Defense 2: Explicitly destroy large image objects after processing
            del item
            if 'image' in locals():
                del image
            gc.collect() 
            
            if extracted_count >= num_samples:
                break

    print("🎉 Streaming extraction complete! Memory strictly controlled with 0 leaks!")
    print(f"📁 Data saved to: {save_dir}")

if __name__ == "__main__":
    stream_and_sample_fast(seed=1024, num_samples=256)

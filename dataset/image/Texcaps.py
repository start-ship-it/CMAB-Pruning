# texcaps.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"

import json
import gc
import sys
from datasets import load_dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = 10_000_000 

def print_checkpoint(step_name):

    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_kb = int(line.split()[1])
                    mem_mb = mem_kb / 1024
                    print(f"[{step_name}] - Current memory usage: {mem_mb:.2f} MB", flush=True)
                    return
    except:
        pass
    print(f"[{step_name}] - Checkpoint passed", flush=True)

def stream_textcaps(seed=0, num_samples=256, output_dir="./textcaps_sampled"):
    dataset_name = "lmms-lab-encoder/TextCaps"
    print(f"🚀 [TextCaps] Starting streaming sampling (Seed={seed})...", flush=True)
    
    print_checkpoint("Step 1: Preparing to call load_dataset (using hf-mirror)")
    
    # Enable streaming to pull only necessary data
    dataset = load_dataset(
        dataset_name, 
        split="val", 
        streaming=True
    )
    
    print_checkpoint("Step 2: load_dataset called successfully, connection established")
    
    # Reduce buffer_size to 20 to prevent OOM
    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=20)
    
    print_checkpoint("Step 3: Shuffle preparation complete")
    
    save_dir = os.path.join(output_dir, f"textcaps_seed{seed}")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"textcaps_val_{num_samples}.jsonl")
    
    print_checkpoint("Step 4: Directories prepared, entering data fetching loop")
    
    extracted_count = 0
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        # Wrap with enumerate to accurately capture fetching progress
        for i, item in enumerate(shuffled_dataset):
            print_checkpoint(f"Successfully fetched underlying stream data for item {i+1}")
            
            try:
                if 'image' not in item or item['image'] is None:
                    continue
                
                # Intercept abnormally large images again
                img_width, img_height = item['image'].size
                if img_width * img_height > 8_000_000:
                    print(f"  [Skipped] Image dimensions too large ({img_width}x{img_height})", flush=True)
                    continue
                    
                image = item['image'].convert("RGB")
                query = "Describe this image in detail."
                
                # Dynamically extract caption field
                label = ""
                for key in ['caption_str', 'caption', 'text', 'answers']:
                    if key in item and item[key]:
                        label = item[key][0] if isinstance(item[key], list) else item[key]
                        break
                        
                label = str(label).strip()
                if not label:
                    continue
                    
                # Save image
                img_filename = f"textcaps_{extracted_count}.png"
                image.save(os.path.join(img_dir, img_filename))
                
                # Save annotation
                record = {
                    "id": f"seed{seed}_{extracted_count}",
                    "image": f"images/{img_filename}",
                    "query": query,
                    "label": label
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush() # Force write to disk
                
                extracted_count += 1
                if extracted_count % 10 == 0 or extracted_count == num_samples:
                    print(f"  ✅ [Progress] Successfully extracted {extracted_count}/{num_samples} samples", flush=True)
                    
                if extracted_count >= num_samples:
                    break
                    
            except Exception as e:
                print(f"  [Skipped] Exception during processing item {i+1}: {e}", flush=True)
                continue
                
            finally:
                # Crucial step: forcibly clear variables and run garbage collection every iteration to prevent memory spikes
                if 'image' in locals():
                    del image
                if 'item' in locals():
                    del item
                gc.collect()
                
    print(f"🎉 [TextCaps] Extraction complete! Data safely saved to: {save_dir}", flush=True)

if __name__ == "__main__":
    stream_textcaps(seed=1024, num_samples=256)

# download_text.py
'''
aria2c -x 16 -s 16 https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -d /root/autodl-tmp/

mv /root/autodl-tmp/d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb /root/autodl-tmp/ShareGPT_V3_unfiltered_cleaned_split.json
'''

import os
import json
import random  # Added: for shuffling native lists
import ijson # Imported extremely low-memory parser
# ==========================================
# 🌟 Network Fix: Enable HF domestic mirror, clear proxies
# ==========================================
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

# ==========================================
# Universal extraction function (suitable for normal-sized datasets)
# ==========================================
def process_text_dataset(dataset_path, config_name, split_name, save_name, formatter_func, filter_func=None, seed=3407, num_samples=500, output_dir="./text_calib_sampled", **kwargs):
    print(f"\n📥 Downloading text dataset [{save_name}] ({dataset_path})...")
    
    # Full-speed offline download to local storage (pass-through **kwargs)
    if config_name:
        dataset = load_dataset(dataset_path, config_name, split=split_name, **kwargs)
    else:
        dataset = load_dataset(dataset_path, split=split_name, **kwargs)
        
    print(f"✅ Download complete! Total {len(dataset)} records.")

    if filter_func:
        print("🧹 Cleaning invalid/too short text data...")
        dataset = dataset.filter(filter_func)

    print(f"🎲 Shuffling with Seed={seed}, extracting {num_samples} samples...")
    sampled_dataset = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))

    save_dir = os.path.join(output_dir, f"{save_name}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"{save_name}_{num_samples}.jsonl")
    
    print(f"💾 Saving to {jsonl_path}...")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(sampled_dataset):
            record = formatter_func(item, seed, i)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"🎉 [{save_name}] Extraction complete!")

# ==========================================
# Formatting extraction logic for different datasets
# ==========================================

def fmt_wikitext(item, seed, i):
    return {"id": f"wikitext_seed{seed}_{i}", "text": item["text"].strip()}

def fmt_alpaca(item, seed, i):
    return {
        "id": f"alpaca_seed{seed}_{i}", 
        "instruction": item.get("instruction", ""),
        "input": item.get("input", ""),
        "output": item.get("output", "")
    }

def fmt_sharegpt(item, seed, i):
    return {"id": f"sharegpt_seed{seed}_{i}", "conversations": item.get("conversations", [])}

if __name__ == "__main__":
    SEED = 1024
    NUM_SAMPLES = 500
    OUTPUT_ROOT = "./"
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 1. WikiText-2
    process_text_dataset(
        dataset_path="wikitext", 
        config_name="wikitext-2-raw-v1", 
        split_name="train", 
        save_name="wikitext2",
        formatter_func=fmt_wikitext,
        filter_func=lambda x: len(x["text"].strip()) > 50,
        seed=SEED, num_samples=NUM_SAMPLES, output_dir=OUTPUT_ROOT
    )

    # 2. Alpaca Cleaned
    process_text_dataset(
        dataset_path="yahma/alpaca-cleaned", 
        config_name=None, 
        split_name="train", 
        save_name="alpaca",
        formatter_func=fmt_alpaca,
        seed=SEED, num_samples=NUM_SAMPLES, output_dir=OUTPUT_ROOT
    )

    # 3. ShareGPT (🌟 Ultimate solution: ijson streaming parsing + reservoir sampling)
    print("\n📥 Reading ShareGPT using [Streaming Iteration + Reservoir Sampling] (0 memory pressure throughout)...")
    sharegpt_path = "/root/autodl-tmp/ShareGPT_V3_unfiltered_cleaned_split.json"
    
    reservoir = []
    random.seed(SEED)
    
    # Must use 'rb' binary mode for reading, ijson is fastest this way
    with open(sharegpt_path, 'rb') as f:
        # ijson.items creates a generator, only the current 1 record exists in memory at any time
        for i, item in enumerate(ijson.items(f, 'item')):
            # Reservoir sampling algorithm: equiprobably extract NUM_SAMPLES records without knowing total count
            if i < NUM_SAMPLES:
                reservoir.append(item)
            else:
                j = random.randint(0, i)
                if j < NUM_SAMPLES:
                    reservoir[j] = item
                    
    print(f"✅ OOM perfectly avoided! Successfully extracted {NUM_SAMPLES} samples from massive data.")
    
    sharegpt_dir = os.path.join(OUTPUT_ROOT, f"sharegpt_clean_seed{SEED}")
    os.makedirs(sharegpt_dir, exist_ok=True)
    sharegpt_out = os.path.join(sharegpt_dir, f"sharegpt_clean_{NUM_SAMPLES}.jsonl")
    
    print(f"💾 Saving to {sharegpt_out}...")
    with open(sharegpt_out, 'w', encoding='utf-8') as f:
        for i, item in enumerate(reservoir):
            record = fmt_sharegpt(item, SEED, i)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("🎉 [sharegpt_clean] Extraction complete!")
    print("\n✅ All text calibration sets are ready!")

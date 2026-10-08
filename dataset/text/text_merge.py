import os
import json
import random

def merge_and_sample_jsonl(file_paths, output_path, seed=3407, x_per_dataset=256):
    """
    Core logic: Extract the first x samples from each independent dataset, 
    merge the extracted data, and perform a global shuffle before saving.
    """
    print(f"🔄 Starting processing: Extracting the first x={x_per_dataset} samples from each dataset...")
    combined_data = []

    # 1. Read files individually and [extract first]
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        dataset_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset_lines.append(json.loads(line.strip()))
        
        # 🌟 Core logic: Extract the first x samples from the current dataset before merging
        sampled_lines = dataset_lines[:x_per_dataset]
        combined_data.extend(sampled_lines)
        
        print(f"  - Read {len(dataset_lines)} samples from {os.path.basename(file_path)}, kept {len(sampled_lines)} samples.")

    print(f"\n✅ Extraction from all datasets complete! Successfully merged a total of {len(combined_data)} samples.")

    # 2. [Merge and Shuffle] Apply a global random seed to thoroughly shuffle the combined data
    print(f"🎲 Thoroughly shuffling the merged data using global Seed={seed}...")
    random.seed(seed)
    random.shuffle(combined_data)

    # 3. Write to the final JSONL file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"💾 Saving to: {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"🎉 Pure text multi-source mixed calibration set successfully generated!")

if __name__ == "__main__":
    SEED = 1024
    
    # 🌟 Set the number of samples (x) you want to extract from [each dataset] here
    # For example, setting it to 128 means the 3 datasets will be merged into a 3 * 128 = 384 sample mixed dataset
    X_PER_DATASET = 256
    
    BASE_DIR = "/root/autodl-tmp/revise/data/text"
    
    INPUT_FILES = [
        os.path.join(BASE_DIR, f"alpaca_seed{SEED}", "alpaca_500.jsonl"),
        os.path.join(BASE_DIR, f"sharegpt_clean_seed{SEED}", "sharegpt_clean_500.jsonl"),
        os.path.join(BASE_DIR, f"wikitext2_seed{SEED}", "wikitext2_500.jsonl")
    ]
    
    # INPUT_FILES = [
    #     os.path.join(BASE_DIR, "alpaca_seed2026", "alpaca_500.jsonl"),
    #     os.path.join(BASE_DIR, "sharegpt_clean_seed2026", "sharegpt_clean_500.jsonl"),
    #     os.path.join(BASE_DIR, "wikitext2_seed2026", "wikitext2_500.jsonl")
    # ]
    
    # The final output filename will automatically calculate the total number of samples (e.g., 3 * 256 = 768)
    total_samples = X_PER_DATASET * len(INPUT_FILES)
    OUTPUT_FILE = os.path.join(BASE_DIR, f"mixed_text_seed{SEED}_{total_samples}.jsonl")
    
    merge_and_sample_jsonl(
        file_paths=INPUT_FILES,
        output_path=OUTPUT_FILE,
        seed=SEED,
        x_per_dataset=X_PER_DATASET
    )

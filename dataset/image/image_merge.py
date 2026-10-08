# image_merge.py
import os
import json
import random

def merge_and_shuffle(base_dir, input_files, output_json_name, extract_num, global_shuffle_seed=2026):
    print(f"🔄 Step 1: Starting dataset merge (extracting top {extract_num} samples from each dataset)...")
    
    all_data = []
    
    for rel_file_path in input_files:
        full_path = os.path.join(base_dir, rel_file_path)
        
        if not os.path.exists(full_path):
            print(f"⚠️ File not found, skipping: {rel_file_path}")
            continue
            
        dataset_rel_dir = os.path.dirname(rel_file_path)
        dataset_name = dataset_rel_dir.split('/')[0]
        
        count = 0
        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 🌟 Extraction logic: immediately stop reading after reaching the specified extract_num
                if count >= extract_num:
                    break
                    
                if line.strip():
                    record = json.loads(line)
                    
                    # Correct image paths
                    original_img_path = record['image']
                    corrected_img_path = os.path.join(dataset_rel_dir, original_img_path) 
                    record['image'] = corrected_img_path
                    record['source_dataset'] = dataset_name
                    
                    all_data.append(record)
                    count += 1
                    
        print(f"✅ Successfully extracted and corrected {count} samples from {dataset_name}")

    print(f"\n🎲 Step 2: Globally shuffling the aggregated {len(all_data)} samples using random seed (Seed={global_shuffle_seed})...")
    random.seed(global_shuffle_seed)
    random.shuffle(all_data)
    
    output_path = os.path.join(base_dir, output_json_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 [Success] Dataset extraction, merging, and shuffling completed!")
    print(f"💾 Final calibration set saved to: {output_path}")

if __name__ == "__main__":
    BASE_DIR = "./image"
    
    input_jsonl_files = [
        "chartqa_sampled/chartqa_seed2026/chartqa_test_256.jsonl",
        "docvqa_sampled/documentvqa_seed2026/documentvqa_validation_256.jsonl",
        "llava_coco_sampled/llava_coco_seed2026/llava_coco_256.jsonl",
        "scienceqa_sampled/scienceqa_seed2026/scienceqa_val_256.jsonl",
        "sharegpt4v_sampled/llava_instruct_seed2026/llava_instruct_256.jsonl",
        "textcaps_sampled/textcaps_seed2026/textcaps_val_256.jsonl" 
    ]
    
    # Set the number of samples to extract per dataset
    x = 32
    
    OUTPUT_FILE = f"mixed_calibration_{6*x}.json"
    
    merge_and_shuffle(
        base_dir=BASE_DIR, 
        input_files=input_jsonl_files, 
        output_json_name=OUTPUT_FILE, 
        extract_num=x, 
        global_shuffle_seed=2026  # Set the global shuffling seed
    )

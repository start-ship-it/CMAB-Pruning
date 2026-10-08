# ScienceQA.py
import json
import os
from datasets import load_dataset

def stream_scienceqa(seed=42, num_samples=256, output_dir="./scienceqa_sampled"):
    dataset_name = "derek-thomas/ScienceQA" 
    print(f"🚀 [ScienceQA] Starting streaming sampling (Seed={seed})...")
    
    dataset = load_dataset(dataset_name, split="validation", streaming=True)
    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    save_dir = os.path.join(output_dir, f"scienceqa_seed{seed}")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"scienceqa_val_{num_samples}.jsonl")
    
    extracted_count = 0
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in shuffled_dataset:
            # 1. Filter pure text data (discard questions without images)
            if 'image' not in item or item['image'] is None:
                continue
                
            try:
                image = item['image'].convert("RGB")
            except:
                continue
                
            # 2. Construct multiple-choice Prompt
            question = item.get('question', '')
            choices = item.get('choices', [])
            if choices:
                question += " \nChoices: " + ", ".join([str(c) for c in choices])
            
            # 3. Parse the ground-truth answer
            ans_idx = item.get('answer', 0)
            if isinstance(choices, list) and len(choices) > int(ans_idx):
                label = str(choices[int(ans_idx)])
            else:
                label = str(ans_idx)
                
            # 4. Save to disk
            img_filename = f"scienceqa_{extracted_count}.png"
            image.save(os.path.join(img_dir, img_filename))
            
            record = {
                "id": f"seed{seed}_{extracted_count}",
                "image": f"images/{img_filename}",
                "query": question,
                "label": label
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            extracted_count += 1
            if extracted_count >= num_samples:
                break
                
    print(f"🎉 [ScienceQA] Extraction complete! Data saved to: {save_dir}")

if __name__ == "__main__":
    stream_scienceqa(seed=1024, num_samples=256)

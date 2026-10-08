# ChartQA.py
#source /etc/network_turbo
#unset_network_turbo
import os

os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
from datasets import load_dataset

def download_and_sample_chartqa(seed=42, num_samples=256, output_dir="./chartqa_sampled"):
    print("📥  Downloading the official ChartQA test set via the domestic mirror (hf-mirror.com)...")
    
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
    print(f"✅ The complete test set has been successfully loaded, containing a total of {len(dataset)} records.")

    print(f"🎲 Shuffle using the random seed `Seed={seed}` and draw `{num_samples}` samples...")
    sampled_dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    save_dir = os.path.join(output_dir, f"chartqa_seed{seed}")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    jsonl_path = os.path.join(save_dir, f"chartqa_test_{num_samples}.jsonl")
    
    print("💾 Saving the image and annotation file to the local drive...")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(sampled_dataset):
            image = item['image']
            query = item['query']
            label = item['label']
            
            img_filename = f"chartqa_{i}.png"
            img_path = os.path.join(img_dir, img_filename)
            image.convert("RGB").save(img_path)
            
            record = {
                "id": f"seed{seed}_{i}",
                "image": f"images/{img_filename}",
                "query": query,
                "label": label
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("🎉 Download and extraction complete！")
    print(f"📄 Annotation File: {jsonl_path}")
    print(f"🖼️ Image Index: {img_dir}")
    print(f"📁 Unified Storage Path: {save_dir}")

if __name__ == "__main__":
    download_and_sample_chartqa(seed=1024, num_samples=256)

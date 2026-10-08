#unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
#wget -c --no-check-certificate -O llava_instruct_150k.json "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"

# llava_coco.py
import json
import os
import random
import urllib.request
from PIL import Image

def get_coco_image(img_basename, save_path):

    urls_to_try = [
        f"http://images.cocodataset.org/train2017/{img_basename}",
        f"http://images.cocodataset.org/train2014/{img_basename}",
        f"http://images.cocodataset.org/val2014/{img_basename}"
    ]
    for url in urls_to_try:
        try:
            # Spoof User-Agent to make requests more stable
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                with open(save_path, 'wb') as f:
                    f.write(response.read())
            return True
        except Exception:
            continue
    return False

def build_pure_llava_coco(seed=3407, num_samples=256, output_dir="./llava_coco_sampled"):
    json_path = "llava_instruct_150k.json"
    if not os.path.exists(json_path):
        print(f"❌ Cannot find {json_path}! Please make sure to run the wget command to download the JSON first.")
        return
        
    print(f"🚀 [Pure LLaVA-COCO] Starting fast on-demand fetching (Seed={seed})...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    valid_data = [item for item in data if 'image' in item]
    print(f"📊 Successfully loaded official LLaVA JSON, containing {len(valid_data)} image-text annotations.")
    
    random.seed(seed)
    random.shuffle(valid_data)
    
    save_dir = os.path.join(output_dir, f"llava_coco_seed{seed}")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"llava_coco_{num_samples}.jsonl")
    
    extracted_count = 0
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in valid_data:
            # Parse image basename
            img_basename = os.path.basename(item['image'])
            
            # Extract QA conversations
            conversations = item.get('conversations', [])
            if len(conversations) < 2:
                continue
                
            query = conversations[0].get('value', '').replace('<image>', '').replace('\n\n', '\n').strip()
            label = conversations[1].get('value', '').strip()
            if not query or not label:
                continue
                
            local_img_path = os.path.join(img_dir, f"llava_coco_{extracted_count}.jpg")
            success = get_coco_image(img_basename, local_img_path)
            
            if not success:
                continue
                
            try:
                # Verify image validity and ensure RGB format (prevents pruning code errors caused by single-channel grayscale images)
                with Image.open(local_img_path) as img:
                    img.convert("RGB")
            except Exception:
                os.remove(local_img_path)
                continue
                
            record = {
                "id": f"seed{seed}_{extracted_count}",
                "image": f"images/llava_coco_{extracted_count}.jpg",
                "query": query,
                "label": label
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            extracted_count += 1
            if extracted_count % 20 == 0 or extracted_count == num_samples:
                print(f"  [Progress] Successfully downloaded and extracted {extracted_count}/{num_samples} samples on demand")
                
            if extracted_count >= num_samples:
                break
                
    print(f"🎉 [Success] The purest LLaVA-COCO calibration set is ready. Data stored in: {save_dir}")

if __name__ == "__main__":
    build_pure_llava_coco(seed=1024, num_samples=256)

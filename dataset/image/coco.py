#unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
#wget -c --no-check-certificate -O llava_instruct_150k.json "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"

# llava_coco.py
import json
import os
import random
import urllib.request
from PIL import Image

def get_coco_image(img_basename, save_path):
    # COCO 官方开放服务器地址 (无防护，无限制，无需代理)
    # 兼容 LLaVA 的多版本图片命名格式
    urls_to_try = [
        f"http://images.cocodataset.org/train2017/{img_basename}",
        f"http://images.cocodataset.org/train2014/{img_basename}",
        f"http://images.cocodataset.org/val2014/{img_basename}"
    ]
    for url in urls_to_try:
        try:
            # 伪装一下 User-Agent 让请求更稳定
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
        print(f"❌ 找不到 {json_path}！请务必先执行 wget 命令下载 JSON。")
        return
        
    print(f"🚀 [纯正 LLaVA-COCO] 开启按需极速拉取 (Seed={seed})...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    valid_data = [item for item in data if 'image' in item]
    print(f"📊 成功读取 LLaVA 官方 JSON，包含 {len(valid_data)} 条图文标注。")
    
    # 根据 Seed 进行洗牌，保证随机性
    random.seed(seed)
    random.shuffle(valid_data)
    
    save_dir = os.path.join(output_dir, f"llava_coco_seed{seed}")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"llava_coco_{num_samples}.jsonl")
    
    extracted_count = 0
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in valid_data:
            # 解析图片名
            img_basename = os.path.basename(item['image'])
            
            # 提取 QA 对话
            conversations = item.get('conversations', [])
            if len(conversations) < 2:
                continue
                
            query = conversations[0].get('value', '').replace('<image>', '').replace('\n\n', '\n').strip()
            label = conversations[1].get('value', '').strip()
            if not query or not label:
                continue
                
            # 🌟 核心魔法：只下载这 1 张图片！免去下载整个 18GB 压缩包的痛苦
            local_img_path = os.path.join(img_dir, f"llava_coco_{extracted_count}.jpg")
            success = get_coco_image(img_basename, local_img_path)
            
            if not success:
                continue
                
            try:
                # 验证图片是否有效，并保证是 RGB 格式（防止单通道黑白图引起剪枝代码报错）
                with Image.open(local_img_path) as img:
                    img.convert("RGB")
            except Exception:
                os.remove(local_img_path)
                continue
                
            # 保存到 jsonl
            record = {
                "id": f"seed{seed}_{extracted_count}",
                "image": f"images/llava_coco_{extracted_count}.jpg",
                "query": query,
                "label": label
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            extracted_count += 1
            if extracted_count % 20 == 0 or extracted_count == num_samples:
                print(f"  [进度] 成功按需下载并提取 {extracted_count}/{num_samples} 条数据")
                
            if extracted_count >= num_samples:
                break
                
    print(f"🎉 [大功告成] 最纯正的 LLaVA-COCO 校准集已准备完毕，数据存放在: {save_dir}")

if __name__ == "__main__":
    build_pure_llava_coco(seed=1024, num_samples=256)
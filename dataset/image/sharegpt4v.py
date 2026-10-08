# source /etc/network_turbo

# wget -c --no-check-certificate -O llava_data.parquet "https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft/resolve/main/data/train-00000-of-00020.parquet"
# share.py
import json
import os
import io
from PIL import Image as PILImage
from datasets import Dataset

# 🌟 新增：无坚不摧的文本提取函数
def safe_extract_text(content):
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                # 如果有 text 字段并且不为 None，才转成字符串
                t = item.get('text')
                if t is not None:
                    texts.append(str(t))
            elif isinstance(item, str):
                texts.append(item)
        return "".join(texts)
    return str(content)

def local_extract(seed=3407, num_samples=256, output_dir="./sharegpt4v_sampled"):
    local_file = "llava_data.parquet"
    
    if not os.path.exists(local_file):
        print(f"❌ 找不到 {local_file}！")
        return

    print("🚀 [LLaVA-Instruct-VSFT] 开启纯本地 SSD 极速提取...")
    dataset = Dataset.from_parquet(local_file)
    shuffled_dataset = dataset.shuffle(seed=seed)
    
    save_dir = os.path.join(output_dir, f"llava_instruct_seed{seed}")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"llava_instruct_{num_samples}.jsonl")
    
    extracted_count = 0
    skip_reasons = {"no_messages": 0, "no_image": 0, "error": 0}
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(shuffled_dataset):
            messages = item.get('messages', [])
            if not messages or len(messages) < 2:
                skip_reasons["no_messages"] += 1
                continue
                
            try:
                # 1. 提取真实图片
                image_data = None
                if 'images' in item and item['images']:
                    image_data = item['images'][0]
                elif 'image' in item and item['image']:
                    image_data = item['image']
                    
                if image_data is None:
                    skip_reasons["no_image"] += 1
                    continue
                    
                if isinstance(image_data, PILImage.Image):
                    image = image_data
                elif isinstance(image_data, dict) and 'bytes' in image_data and image_data['bytes']:
                    image = PILImage.open(io.BytesIO(image_data['bytes']))
                else:
                    raise ValueError("无法解析的图片格式")
                    
                image = image.convert("RGB")
                
                # 2. 🌟 提取对话 (使用刚加的万能清洗函数)
                query = safe_extract_text(messages[0].get('content')).replace('<image>', '').strip()
                label = safe_extract_text(messages[1].get('content')).strip()
                
                # 过滤掉清洗后没有文字的脏数据
                if not query or not label:
                    continue

            except Exception as e:
                if skip_reasons["error"] == 0:
                    print(f"\n❌ [DEBUG] 发现异常数据，跳过原因: {e}")
                skip_reasons["error"] += 1
                continue
                
            # 3. 正常保存
            img_filename = f"llava_instruct_{extracted_count}.png"
            image.save(os.path.join(img_dir, img_filename))
            
            record = {
                "id": f"seed{seed}_{extracted_count}",
                "image": f"images/{img_filename}",
                "query": query,
                "label": label
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            extracted_count += 1
            if extracted_count % 50 == 0 or extracted_count == num_samples:
                print(f"  [进度] 已成功获取 {extracted_count}/{num_samples} 条真实图文数据")
                
            if extracted_count >= num_samples:
                break
                
    print(f"\n📊 --- 提取诊断报告 ---")
    print(f"🎯 成功提取: {extracted_count} 条")
    print(f"⚠️ 因无对话跳过: {skip_reasons['no_messages']} 条")
    print(f"⚠️ 因无图片跳过: {skip_reasons['no_image']} 条")
    print(f"🚨 因格式异常跳过: {skip_reasons['error']} 条")
    print(f"-----------------------\n")
    print(f"🎉 [大功告成] 数据已安全保存至: {save_dir}")

if __name__ == "__main__":
    local_extract(seed=1024, num_samples=256)
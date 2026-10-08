# ChartQA.py
#source /etc/network_turbo
#unset_network_turbo
import os

# ==========================================
# 🌟 修复网络：彻底清除失效代理，启用 HF 国内镜像
# ==========================================
# 1. 清空可能存在的代理环境变量，防止 Connection refused
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# 2. 启用 Hugging Face 国内高速镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
from datasets import load_dataset

def download_and_sample_chartqa(seed=42, num_samples=256, output_dir="./chartqa_sampled"):
    print("📥 正在通过国内镜像 (hf-mirror.com) 下载 ChartQA 官方测试集...")
    
    # split="test" 划分
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
    print(f"✅ 成功加载完整测试集，共 {len(dataset)} 条数据。")

    print(f"🎲 使用随机种子 Seed={seed} 进行打乱，抽取 {num_samples} 条样本...")
    sampled_dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    # 🌟 核心修改：同步 COCO/LLaVA 目录结构格式
    save_dir = os.path.join(output_dir, f"chartqa_seed{seed}")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    jsonl_path = os.path.join(save_dir, f"chartqa_test_{num_samples}.jsonl")
    
    print("💾 正在将图片与标注文件保存至本地...")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(sampled_dataset):
            image = item['image']
            query = item['query']
            label = item['label']
            
            img_filename = f"chartqa_{i}.png"
            img_path = os.path.join(img_dir, img_filename)
            image.convert("RGB").save(img_path)
            
            # 🌟 核心修改：同步 JSONL 内部图片路径格式
            record = {
                "id": f"seed{seed}_{i}",
                "image": f"images/{img_filename}",
                "query": query,
                "label": label
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("🎉 下载与抽取完成！")
    print(f"📄 标注文件: {jsonl_path}")
    print(f"🖼️ 图片目录: {img_dir}")
    print(f"📁 统一存储路径: {save_dir}")

if __name__ == "__main__":
    download_and_sample_chartqa(seed=1024, num_samples=256)
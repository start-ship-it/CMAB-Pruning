# download_text.py
'''
aria2c -x 16 -s 16 https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -d /root/autodl-tmp/

mv /root/autodl-tmp/d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb /root/autodl-tmp/ShareGPT_V3_unfiltered_cleaned_split.json
'''

import os
import json
import random  # 新增：用于原生列表的打乱
import ijson # 引入极低内存解析器
# ==========================================
# 🌟 修复网络：启用 HF 国内镜像，清除代理
# ==========================================
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

# ==========================================
# 通用提取函数 (适用于正常大小的数据集)
# ==========================================
def process_text_dataset(dataset_path, config_name, split_name, save_name, formatter_func, filter_func=None, seed=3407, num_samples=500, output_dir="./text_calib_sampled", **kwargs):
    print(f"\n📥 正在下载文本集 [{save_name}] ({dataset_path})...")
    
    # 全速离线下载到本地 (透传 **kwargs)
    if config_name:
        dataset = load_dataset(dataset_path, config_name, split=split_name, **kwargs)
    else:
        dataset = load_dataset(dataset_path, split=split_name, **kwargs)
        
    print(f"✅ 下载完成！共 {len(dataset)} 条数据。")

    if filter_func:
        print("🧹 正在清洗无效/过短的文本数据...")
        dataset = dataset.filter(filter_func)

    print(f"🎲 使用 Seed={seed} 进行打乱，抽取 {num_samples} 条样本...")
    sampled_dataset = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))

    save_dir = os.path.join(output_dir, f"{save_name}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"{save_name}_{num_samples}.jsonl")
    
    print(f"💾 正在保存至 {jsonl_path}...")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(sampled_dataset):
            record = formatter_func(item, seed, i)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"🎉 [{save_name}] 抽取完毕！")

# ==========================================
# 针对不同数据集的格式化提取逻辑
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

    # 3. ShareGPT (🌟 终极方案：ijson 流式解析 + 蓄水池抽样)
    print("\n📥 正在使用【流式迭代 + 蓄水池抽样】读取 ShareGPT (全程 0 内存压力)...")
    sharegpt_path = "/root/autodl-tmp/ShareGPT_V3_unfiltered_cleaned_split.json"
    
    reservoir = []
    random.seed(SEED)
    
    # 必须使用 'rb' 二进制模式读取，ijson 速度最快
    with open(sharegpt_path, 'rb') as f:
        # ijson.items 会产生生成器，内存中永远只存在当前这 1 条数据
        for i, item in enumerate(ijson.items(f, 'item')):
            # 蓄水池抽样算法：在不知道总数的情况下，等概率随机抽出 NUM_SAMPLES 条
            if i < NUM_SAMPLES:
                reservoir.append(item)
            else:
                j = random.randint(0, i)
                if j < NUM_SAMPLES:
                    reservoir[j] = item
                    
    print(f"✅ 完美规避 OOM！已成功从海量数据中抽出 {NUM_SAMPLES} 条样本。")
    
    sharegpt_dir = os.path.join(OUTPUT_ROOT, f"sharegpt_clean_seed{SEED}")
    os.makedirs(sharegpt_dir, exist_ok=True)
    sharegpt_out = os.path.join(sharegpt_dir, f"sharegpt_clean_{NUM_SAMPLES}.jsonl")
    
    print(f"💾 正在保存至 {sharegpt_out}...")
    with open(sharegpt_out, 'w', encoding='utf-8') as f:
        for i, item in enumerate(reservoir):
            record = fmt_sharegpt(item, SEED, i)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("🎉 [sharegpt_clean] 抽取完毕！")
    print("\n✅ 所有文本校准集已就绪！")
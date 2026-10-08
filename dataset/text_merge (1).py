import os
import json
import random

def merge_and_sample_jsonl(file_paths, output_path, seed=3407, x_per_dataset=256):
    """
    核心逻辑：先在每个独立数据集中截取前 x 条，然后将截取后的数据合并，最后全局打乱保存。
    """
    print(f"🔄 开始处理，正在从每个数据集单独截取前 x={x_per_dataset} 条数据...")
    combined_data = []

    # 1. 逐个读取，并【先截取】
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到文件: {file_path}")
            continue
            
        dataset_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset_lines.append(json.loads(line.strip()))
        
        # 🌟 核心逻辑：在合并前，先针对当前数据集截取前 x 条
        sampled_lines = dataset_lines[:x_per_dataset]
        combined_data.extend(sampled_lines)
        
        print(f"  - 从 {os.path.basename(file_path)} 总读取 {len(dataset_lines)} 条，截取保留 {len(sampled_lines)} 条。")

    print(f"\n✅ 各数据集截取完毕！当前已合并总计 {len(combined_data)} 条数据。")

    # 2. 【后合并打乱】统一随机种子进行全局洗牌
    print(f"🎲 使用全局 Seed={seed} 对合并后的数据进行充分打乱...")
    random.seed(seed)
    random.shuffle(combined_data)

    # 3. 写入最终的 JSONL 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"💾 正在保存至: {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"🎉 纯文本多源混合校准集生成完毕！")

if __name__ == "__main__":
    SEED = 1024
    
    # 🌟 在这里设置您想要从【每个数据集】中截取的数量 (x)
    # 比如设置为 128，那么最终 3 个数据集将合并成 3 * 128 = 384 条的混合数据集
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
    
    
    # 最终输出文件名会自动计算总条数 (例如 3 * 256 = 768)
    total_samples = X_PER_DATASET * len(INPUT_FILES)
    OUTPUT_FILE = os.path.join(BASE_DIR, f"mixed_text_seed{SEED}_{total_samples}.jsonl")
    
    merge_and_sample_jsonl(
        file_paths=INPUT_FILES,
        output_path=OUTPUT_FILE,
        seed=SEED,
        x_per_dataset=X_PER_DATASET
    )
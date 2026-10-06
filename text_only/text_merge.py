import json
import random
import os
from datasets import load_from_disk

def merge_datasets():
    # 定义所有输入路径
    alpaca_path = "/root/autodl-tmp/data/alpaca_calib_256.jsonl"
    sharegpt_path = "/root/autodl-tmp/data/sharegpt_clean_256.json"
    wiki_dir = "/root/autodl-tmp/data/wikitext-2"
    
    output_path = "/root/autodl-tmp/data/unified_text_calib_768_final.jsonl"
    
    final_samples = []

    # 1. 处理 Alpaca (256条)
    print("⏳ 正在读取 Alpaca 数据...")
    if os.path.exists(alpaca_path):
        with open(alpaca_path, "r", encoding="utf-8") as f:
            for line in f:
                final_samples.append(json.loads(line.strip()))
        print(f"✅ 已加载 Alpaca: {len(final_samples)} 条")
    else:
        print(f"❌ 未找到 Alpaca 文件: {alpaca_path}")

    # 2. 处理之前提取好的纯净 ShareGPT (256条)
    print("⏳ 正在读取 ShareGPT 数据并格式化...")
    if os.path.exists(sharegpt_path):
        with open(sharegpt_path, "r", encoding="utf-8") as f:
            sharegpt_data = json.load(f)
            for i, item in enumerate(sharegpt_data):
                # 将对话列表转化为标准文本格式
                conv_text = ""
                for turn in item.get("conversations", []):
                    role = "User: " if turn["from"] == "human" else "Assistant: "
                    conv_text += f"{role}{turn['value'].strip()}\n\n"
                
                final_samples.append({
                    "id": f"sharegpt_{i}",
                    "text": conv_text.strip(),
                    "source": "sharegpt"
                })
        print(f"✅ 已加载并格式化 ShareGPT: 256 条")

    # 3. 处理 Wikitext-2 (提取 256 条)
    print("⏳ 正在从本地磁盘加载 Wikitext-2 并筛选优质样本...")
    try:
        wiki_ds = load_from_disk(wiki_dir)
        # 筛选条件：长度大于 200 字符，且不是标题（不以 = 开头）
        valid_wiki = [
            item["text"].strip() for item in wiki_ds 
            if len(item["text"].strip()) > 200 and not item["text"].strip().startswith("=")
        ]
        
        random.seed(42)
        wiki_sampled = random.sample(valid_wiki, 256)
        
        for i, text in enumerate(wiki_sampled):
            final_samples.append({
                "id": f"wikitext_{i}",
                "text": text,
                "source": "wikitext"
            })
        print(f"✅ 已提取并加载 Wikitext-2: 256 条")
    except Exception as e:
        print(f"❌ 处理 Wikitext 失败: {e}")

    # 4. 最终合并与保存
    print(f"⏳ 正在合并并保存至 {output_path}...")
    random.shuffle(final_samples) # 打乱顺序，让校准更均匀
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"🎉 恭喜！全模态纯净校准集构建完成，共 {len(final_samples)} 条。")

if __name__ == "__main__":
    merge_datasets()
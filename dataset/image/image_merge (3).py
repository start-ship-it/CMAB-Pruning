# image_merge.py
import os
import json
import random

def merge_and_shuffle(base_dir, input_files, output_json_name, extract_num, global_shuffle_seed=2026):
    print(f"🔄 第一步：开始合并数据集 (每个数据集截取前 {extract_num} 条)...")
    
    all_data = []
    
    # 1. 遍历并截取所有子数据集
    for rel_file_path in input_files:
        full_path = os.path.join(base_dir, rel_file_path)
        
        if not os.path.exists(full_path):
            print(f"⚠️ 找不到文件，已跳过: {rel_file_path}")
            continue
            
        dataset_rel_dir = os.path.dirname(rel_file_path)
        dataset_name = dataset_rel_dir.split('/')[0]
        
        count = 0
        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 🌟 截取逻辑：达到设定的抽取数量 extract_num 后，立即停止读取
                if count >= extract_num:
                    break
                    
                if line.strip():
                    record = json.loads(line)
                    
                    # 修正图片路径
                    original_img_path = record['image']
                    corrected_img_path = os.path.join(dataset_rel_dir, original_img_path) 
                    record['image'] = corrected_img_path
                    record['source_dataset'] = dataset_name
                    
                    all_data.append(record)
                    count += 1
                    
        print(f"✅ 从 {dataset_name} 成功截取并修正了 {count} 条数据")

    # 2. 🌟 核心打乱逻辑：在截取的基础上，设置随机种子进行全局打乱
    print(f"\n🎲 第二步：正在使用随机种子 (Seed={global_shuffle_seed}) 对汇总的 {len(all_data)} 条数据进行全局随机打乱...")
    random.seed(global_shuffle_seed)
    random.shuffle(all_data)
    
    # 3. 导出保存
    output_path = os.path.join(base_dir, output_json_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 [大功告成] 数据集截取与混合打乱完毕！")
    print(f"💾 最终校准集已保存至: {output_path}")

if __name__ == "__main__":
    BASE_DIR = "/root/autodl-tmp/revise/data/image"
    
    input_jsonl_files = [
        "chartqa_sampled/chartqa_seed2026/chartqa_test_256.jsonl",
        #"docvqa_sampled/documentvqa_seed2026/documentvqa_validation_256.jsonl",
        "llava_coco_sampled/llava_coco_seed2026/llava_coco_256.jsonl",
        "scienceqa_sampled/scienceqa_seed2026/scienceqa_val_256.jsonl",
        "sharegpt4v_sampled/llava_instruct_seed2026/llava_instruct_256.jsonl",
        "textcaps_sampled/textcaps_seed2026/textcaps_val_256.jsonl" 
    ]
    
    # 设置每个数据集截取的数量
    x = 32
    
    # 动态生成输出文件名 (例如 mixed_calibration_192.json)
    OUTPUT_FILE = f"mixed_calibration_{6*x}.json"
    
    merge_and_shuffle(
        base_dir=BASE_DIR, 
        input_files=input_jsonl_files, 
        output_json_name=OUTPUT_FILE, 
        extract_num=x, 
        global_shuffle_seed=2026  # 设定全局打乱的随机种子
    )
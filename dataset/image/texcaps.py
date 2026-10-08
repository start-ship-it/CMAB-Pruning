# texcaps.py
import os

# ⚠️ 核心网络修复：在导入任何 HF 库之前，强制指定使用国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 防止 datasets 在后台疯狂缓存吃爆内存
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"

import json
import gc
import sys
from datasets import load_dataset
from PIL import Image

# ⚠️ 核心内存修复：防止“图片炸弹”耗尽内存直接导致 Killed
Image.MAX_IMAGE_PIXELS = 10_000_000 

def print_checkpoint(step_name):
    """内存探针，用于实时监控程序死在哪里"""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_kb = int(line.split()[1])
                    mem_mb = mem_kb / 1024
                    print(f"[{step_name}] - 当前内存占用: {mem_mb:.2f} MB", flush=True)
                    return
    except:
        pass
    print(f"[{step_name}] - 检查点通过", flush=True)

def stream_textcaps(seed=0, num_samples=256, output_dir="./textcaps_sampled"):
    dataset_name = "lmms-lab-encoder/TextCaps"
    print(f"🚀 [TextCaps] 开启流式采样 (Seed={seed})...", flush=True)
    
    print_checkpoint("步骤 1: 准备调用 load_dataset (使用 hf-mirror 镜像源)")
    
    # 开启流式加载，仅拉取需要的数据
    dataset = load_dataset(
        dataset_name, 
        split="val", 
        streaming=True
    )
    
    print_checkpoint("步骤 2: load_dataset 调用成功，连接已建立")
    
    # 将 buffer_size 降到 20，防 OOM
    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=20)
    
    print_checkpoint("步骤 3: shuffle 准备完毕")
    
    save_dir = os.path.join(output_dir, f"textcaps_seed{seed}")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"textcaps_val_{num_samples}.jsonl")
    
    print_checkpoint("步骤 4: 目录准备完毕，进入数据拉取循环")
    
    extracted_count = 0
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        # 使用 enumerate 包装，精确捕捉拉取进度
        for i, item in enumerate(shuffled_dataset):
            print_checkpoint(f"成功拉取到底层流数据 第 {i+1} 条")
            
            try:
                if 'image' not in item or item['image'] is None:
                    continue
                
                # 再次拦截异常大图
                img_width, img_height = item['image'].size
                if img_width * img_height > 8_000_000:
                    print(f"  [跳过] 图片尺寸过大 ({img_width}x{img_height})", flush=True)
                    continue
                    
                image = item['image'].convert("RGB")
                query = "Describe this image in detail."
                
                # 动态提取 caption 字段
                label = ""
                for key in ['caption_str', 'caption', 'text', 'answers']:
                    if key in item and item[key]:
                        label = item[key][0] if isinstance(item[key], list) else item[key]
                        break
                        
                label = str(label).strip()
                if not label:
                    continue
                    
                # 存图
                img_filename = f"textcaps_{extracted_count}.png"
                image.save(os.path.join(img_dir, img_filename))
                
                # 存标注
                record = {
                    "id": f"seed{seed}_{extracted_count}",
                    "image": f"images/{img_filename}",
                    "query": query,
                    "label": label
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush() # 强制写入磁盘
                
                extracted_count += 1
                if extracted_count % 10 == 0 or extracted_count == num_samples:
                    print(f"  ✅ [进度] 已成功提取 {extracted_count}/{num_samples} 条", flush=True)
                    
                if extracted_count >= num_samples:
                    break
                    
            except Exception as e:
                print(f"  [跳过] 处理第 {i+1} 条数据时异常: {e}", flush=True)
                continue
                
            finally:
                # 极其关键：每次循环强制清空变量和垃圾回收，保持内存不暴涨
                if 'image' in locals():
                    del image
                if 'item' in locals():
                    del item
                gc.collect()
                
    print(f"🎉 [TextCaps] 抽取完成！数据已安全保存至: {save_dir}", flush=True)

if __name__ == "__main__":
    # 执行主函数
    stream_textcaps(seed=1024, num_samples=256)
from modelscope import snapshot_download

# 指定模型ID，这里假设为Qwen2.0VL2B对应的ID，若实际不同需替换
model_id = "Qwen/Qwen3-VL-8B-Instruct"
# 指定下载目录，可根据需求修改
cache_dir = "/root/autodl-tmp/CMA-Pruning/data"
model_dir = snapshot_download(model_id, cache_dir=cache_dir)
print(f"模型已下载到：{model_dir}")
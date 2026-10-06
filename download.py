from modelscope import snapshot_download

model_id = "Qwen/Qwen3-VL-8B-Instruct"
cache_dir = "/root/autodl-tmp/CMAB-Pruning"
model_dir = snapshot_download(model_id, cache_dir=cache_dir)
print(f"downlad：{model_dir}")

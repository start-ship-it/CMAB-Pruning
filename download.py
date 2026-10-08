from modelscope import snapshot_download

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
cache_dir = "./models"
model_dir = snapshot_download(model_id, cache_dir=cache_dir)
print(f"downlad：{model_dir}")

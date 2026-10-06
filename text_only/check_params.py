import torch
from transformers import AutoModelForImageTextToText

# 自动复用你的模型路径
model_path = "/root/autodl-tmp/Qwen/Qwen3-VL-8B-Instruct"

print(f"⏳ 正在加载模型结构: {model_path}")
print("   (使用 bfloat16 和 device_map='auto' 极速加载中...)")

# 使用与你推理脚本完全一致的加载方式
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

text_params = 0
vision_params = 0
projector_params = 0

# 针对 Qwen-VL 架构的精确参数分类逻辑
for name, param in model.named_parameters():
    num_params = param.numel()
    
    # 1. 跨模态投影层 (在 Qwen 中通常命名为 visual.merger)
    if "visual.merger" in name:
        projector_params += num_params
        
    # 2. 视觉编码器 (排除 merger 后，剩余的 visual 均属于视觉层)
    elif "visual" in name:
        vision_params += num_params
        
    # 3. 文本基座 (包含 model.embed_tokens, model.layers, model.norm, lm_head 等)
    else:
        text_params += num_params

total_params = text_params + vision_params + projector_params

print("\n" + "="*50)
print("📊 Qwen-VL 架构参数量精准统计")
print("="*50)
print(f"总参数量: {total_params / 1e9:.3f} B")
print(f"📖 文本基座参数: {text_params / 1e9:.3f} B (占比 {text_params/total_params*100:.2f}%)")
print(f"👁️ 视觉编码器参数: {vision_params / 1e6:.2f} M (占比 {vision_params/total_params*100:.2f}%)")
print(f"🌉 跨模态投影层参数: {projector_params / 1e6:.2f} M (占比 {projector_params/total_params*100:.2f}%)")
print("="*50)
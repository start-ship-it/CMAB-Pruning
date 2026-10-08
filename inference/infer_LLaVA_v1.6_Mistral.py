from PIL import Image
import torch
from modelscope import AutoProcessor, LlavaForConditionalGeneration

model_id = "./llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe all the information in the image."},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "./inference/data.png" 
raw_image = Image.open(image_file).convert("RGB")  

inputs = processor(
    images=raw_image,
    text=prompt,
    return_tensors='pt'
).to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

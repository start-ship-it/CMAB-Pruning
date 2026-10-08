import torch
from transformers import LlavaForConditionalGeneration, AutoConfig
from prettytable import PrettyTable 

def analyze_llava_structure(model_path):
    print(f"🔍 Loading the model structure from {model_path}...")
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="meta", 
        torch_dtype=torch.float16
    )

    stats = {
        "Vision Tower (Vision)": 0,
        "Projector (Projection Layer)": 0,
        "Language Model (Text)": 0,
        "Total": 0
    }

    for name, param in model.named_parameters():
        params_count = param.numel()
        stats["Total"] += params_count
        
        if "vision_tower" in name:
            stats["Vision Tower (Vision)"] += params_count
        elif "multi_modal_projector" in name:
            stats["Projector (Projection Layer)"] += params_count
        elif "language_model" in name:
            stats["Language Model (Text)"] += params_count

    table = PrettyTable()
    table.field_names = ["Module Name", "Number of Parameters (Params)", "Percentage (%)"]
    table.align["Module Name"] = "l"
    table.align["Number of Parameters (Params)"] = "r"

    for key in ["Vision Tower (Vision)", "Projector Layer", "Language Model (Text)"]:
        count = stats[key]
        percentage = (count / stats["Total"]) * 100
        table.add_row([key, f"{count:,}", f"{percentage:.2f}%"])

    table.add_row(["-"*20, "-"*15, "-"*10])
    table.add_row(["Total", f"{stats['Total']:,}", "100.00%"])

    print(table)

    if hasattr(model, 'language_model'):

        lm_model = model.language_model.model if hasattr(model.language_model, 'model') else model.language_model
        if hasattr(lm_model, 'layers'):
            num_layers = len(lm_model.layers)
            hidden_size = lm_model.config.hidden_size
            print(f"\n📝 Text Layer Details:")
            print(f"   - Layers: {num_layers}")
            print(f"   - Hidden Size: {hidden_size}")
            print(f"   - Vocab Size: {lm_model.config.vocab_size}")

if __name__ == "__main__":

    MODEL_PATH = "./llava-hf/llava-1.5-7b-hf" 
    analyze_llava_structure(MODEL_PATH)

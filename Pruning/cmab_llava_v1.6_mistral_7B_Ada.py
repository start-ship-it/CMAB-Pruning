import torch
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import time
import os

class MultimodalCalibration:
    @staticmethod
    def get_text_samples(jsonl_path, num_samples=None):
        print(f"📚 Loading text set: {jsonl_path}")
        texts = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if num_samples is not None and len(texts) >= num_samples: break
                    data = json.loads(line.strip())
                    content = data.get('text', "")
                    if not content and 'conversations' in data:
                        content = "\n".join([c['value'] for c in data['conversations'] if c['from'] == 'human'])
                    if len(content.strip()) > 10:
                        texts.append(content.strip())
        except Exception as e: print(f"❌ Read failed: {e}")
        print(f"✅ Loaded successfully {len(texts)} Sample text.")
        return texts

    @staticmethod
    def get_vision_samples(json_path, image_folder, num_samples=None):
        print(f"🖼️ Loading the image and text instruction set: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        samples, image_root = [], Path(image_folder)
        for item in data:
            if num_samples is not None and len(samples) >= num_samples: break
            img_full_path = image_root / item.get('image', '')
            if not img_full_path.exists(): continue
            prompt = "Describe this image in detail."
            if 'conversations' in item:
                for conv in item['conversations']:
                    if conv.get('from') == 'human':
                        prompt = conv.get('value', '').replace('<image>', '').strip()
                        break
            try:
                img = Image.open(img_full_path).convert("RGB")
                samples.append({"image": img, "text": prompt})
            except Exception: continue
        print(f"✅ Loaded successfully {len(samples)} Sample image or text.")
        return samples

class CMAProfiler:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.act_stats = {} 
        self.current_mode = None

    def _get_model_layers(self):
        if hasattr(self.model, 'language_model'):
            llm = self.model.language_model
            if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
                return llm.model.layers
            elif hasattr(llm, 'layers'):
                return llm.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'layers'):
            return self.model.layers
        raise AttributeError("The Transformer layer cannot be located. Please check the model architecture.")

    def register_hooks(self):
        self.remove_hooks()
        def make_hook(name):
            def hook(module, input, output):
                x = input[0].detach()
                dim = x.shape[-1]
                x_flat = x.view(-1, dim)
                sq_sum = torch.sum(x_flat.pow(2), dim=0).cpu().double()
                count = x_flat.shape[0]

                if name not in self.act_stats:
                    self.act_stats[name] = {
                        't_sq': torch.zeros(dim, dtype=torch.float64), 't_cnt': 0,
                        'v_sq': torch.zeros(dim, dtype=torch.float64), 'v_cnt': 0
                    }
                if self.current_mode == 'txt':
                    self.act_stats[name]['t_sq'] += sq_sum
                    self.act_stats[name]['t_cnt'] += count
                else:
                    self.act_stats[name]['v_sq'] += sq_sum
                    self.act_stats[name]['v_cnt'] += count
            return hook

        layers = self._get_model_layers()
        for i, layer in enumerate(layers):
            for n, m in layer.named_modules():
                if isinstance(m, nn.Linear):
                    self.hooks.append(m.register_forward_hook(make_hook(f"layer_{i}_{n}")))

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def get_cma_norms(self, name, sparsity, omega=1.0, tau_0=1.0, gamma=1):

        s = self.act_stats[name]
        eps = 1e-8
        a_t = torch.sqrt(s['t_sq'] / s['t_cnt']) if s['t_cnt'] > 0 else torch.zeros_like(s['t_sq'])
        a_v = torch.sqrt(s['v_sq'] / s['v_cnt']) if s['v_cnt'] > 0 else torch.zeros_like(s['v_sq'])
        a_t, a_v = a_t.float(), a_v.float()
        log_t = torch.log1p(a_t)
        log_v = torch.log1p(a_v)
        norm_t = log_t / (torch.max(log_t) + eps)
        norm_v = log_v / (torch.max(log_v) + eps)
        p_t = norm_t / (torch.sum(norm_t) + eps)
        p_v = norm_v / (torch.sum(norm_v) + eps)
        energy_t = torch.sum(norm_t.pow(2))
        energy_v = torch.sum(norm_v.pow(2))
        E_l = torch.log((energy_v + eps) / (energy_t + eps))
        var_t = torch.var(norm_t)
        var_v = torch.var(norm_v)
        delta_sigma_l = (var_v - var_t) / (var_v + var_t + eps)
        M = 0.5 * (p_v + p_t)
        kl_v = torch.sum(p_v * torch.log((p_v + eps) / (M + eps)))
        kl_t = torch.sum(p_t * torch.log((p_t + eps) / (M + eps)))
        js_div = 0.5 * kl_v + 0.5 * kl_t
        G_l = E_l + omega * delta_sigma_l * js_div
        tau = tau_0 * ((1.0 - sparsity) ** gamma)
        tau = max(tau, 1e-4) 
        lambda_l = 1.0 / (1.0 + torch.exp(-G_l / tau))
        act_total = lambda_l * norm_v + (1.0 - lambda_l) * norm_t
        return act_total, lambda_l.item()

class LlavaCmaPruner:
    def __init__(self, model_path, sparsity=0.5, omega=1.0, tau_0=1.0, gamma=1):
        self.model_path = model_path
        self.sparsity = sparsity
        self.omega = omega
        self.tau_0 = tau_0
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.profiler = None
        
        self.detailed_lambdas = {}
        self.layer_lambdas_avg = []

    def load_model(self):
        print(f"📦 Loading ... LLaVA-v1.6-Mistral-7B: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"
        ).eval()

    def run_calibration(self, txt_p, vis_j, vis_i, num=None):
        t_samples = MultimodalCalibration.get_text_samples(txt_p, num)
        v_samples = MultimodalCalibration.get_vision_samples(vis_j, vis_i, num)

        self.profiler = CMAProfiler(self.model)
        self.profiler.register_hooks()

        self.profiler.current_mode = 'txt'
        for text in tqdm(t_samples, desc="Processing Text"):
            prompt = f"[INST] USER: {text}\nASSISTANT: [/INST]"
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad(): self.model(**inputs)

        self.profiler.current_mode = 'vis'
        for s in tqdm(v_samples, desc="Processing Vision"):
            prompt = f"[INST] USER: <image>\n{s['text']}\nASSISTANT: [/INST]"
            inputs = self.processor(text=prompt, images=s['image'], return_tensors="pt").to(self.device)
            with torch.no_grad(): self.model(**inputs)
            del inputs; torch.cuda.empty_cache()

        self.profiler.remove_hooks()

    def save_calibration_stats(self, save_path):
        print(f"💾 Save statistics to: {save_path}")
        torch.save(self.profiler.act_stats, save_path)

    def load_calibration_stats(self, load_path):
        print(f"🔄 Cache Load Statistics: {load_path}")
        if self.profiler is None: self.profiler = CMAProfiler(self.model)
        self.profiler.act_stats = torch.load(load_path, map_location="cpu", weights_only=False)

    def execute_pruning(self):
        print(f"✂️ Execute IT-CMG Pure Version Adaptive Pruning (Sparsity={self.sparsity})...")
        layers = self.profiler._get_model_layers()
        
        self.detailed_lambdas = {}
        self.layer_lambdas_avg = []
        attn_lambdas = []
        mlp_lambdas = []
        
        for i, layer in tqdm(enumerate(layers), total=len(layers), desc="Pruning Layers"):
            layer_lambdas = []
            for n, m in layer.named_modules():
                if isinstance(m, nn.Linear):
                    name = f"layer_{i}_{n}"
                    if name in self.profiler.act_stats:
                        a_cma, dyn_lambda = self.profiler.get_cma_norms(
                            name=name, 
                            sparsity=self.sparsity, 
                            omega=self.omega, 
                            tau_0=self.tau_0, 
                            gamma=self.gamma
                        )
                        a_cma = a_cma.to(m.weight.device)
                        
                        self.detailed_lambdas[name] = float(dyn_lambda)
                        layer_lambdas.append(dyn_lambda)
                        
                        if any(x in n for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                            attn_lambdas.append(dyn_lambda)
                        elif any(x in n for x in ["gate_proj", "up_proj", "down_proj"]):
                            mlp_lambdas.append(dyn_lambda)
                            
                        s_score = torch.abs(m.weight.data) * a_cma.unsqueeze(0)
                        
                        k = int(self.sparsity * s_score.shape[-1])
                        if k > 0:
                            thresholds = torch.sort(s_score, dim=-1)[0][:, k].unsqueeze(-1)
                            mask = (s_score >= thresholds).to(m.weight.dtype)
                            m.weight.data.mul_(mask)
                            
            if layer_lambdas:
                self.layer_lambdas_avg.append(float(np.mean(layer_lambdas)))

        print("\n📊 =============== IT-CMG Dynamic Balanced Routing Final Numerical Report ================")
        if self.layer_lambdas_avg:
            print(f"📈 Global Mean Lambda: {np.mean(self.layer_lambdas_avg):.4f}")
            if attn_lambdas: print(f"🧩 Attn Blocks Mean Lambda: {np.mean(attn_lambdas):.4f}")
            if mlp_lambdas: print(f"🧬 MLP Blocks Mean Lambda: {np.mean(mlp_lambdas):.4f}")
            print("-" * 76)
            mid_idx = len(self.layer_lambdas_avg) // 2
            print(f"  - Shallow Network (Layers 0-{mid_idx-1}) average Lambda: {np.mean(self.layer_lambdas_avg[:mid_idx]):.4f}")
            print(f"  - Deep Network (Layers {mid_idx}-{len(self.layer_lambdas_avg)-1}) average Lambda: {np.mean(self.layer_lambdas_avg[mid_idx:]):.4f}")
        print("===========================================================================\n")

    def save(self, path):
        print(f"💾 Save the pruning model to: {path}")
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)
        
        tokenizer_config_path = os.path.join(path, 'tokenizer_config.json')
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            config_data['use_fast'] = False
            config_data['chat_template'] = "{% for message in messages %}{% if message['role'] == 'user' %}[INST] USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>\n{% endif %}{% endfor %}{% if message['role'] == 'user' %}\n{% else %} [/INST]{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"
            with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
            
        tokenizer_json_path = os.path.join(path, 'tokenizer.json')
        if os.path.exists(tokenizer_json_path):
            os.remove(tokenizer_json_path)
            
        with open(save_dir / "cma_config.json", "w") as f:
            json.dump({
                "sparsity": self.sparsity, 
                "gating_mode": "Pure Information-Theoretic Adaptive",
                "omega": self.omega,
                "tau_0": self.tau_0,
                "gamma": self.gamma,
            }, f, indent=4)
            
        with open(save_dir / "cma_lambda_trajectory.json", "w") as f:
            json.dump({
                "layer_wise_averages": self.layer_lambdas_avg,
                "detailed_linear_lambdas": self.detailed_lambdas
            }, f, indent=4)
            
        print("🎉 The model and adaptive statistics have been successfully saved!")

if __name__ == "__main__":
    SEED = 1024
    CONFIG = {
        "model": "/llava-hf/llava-v1.6-mistral-7b-hf", 
        "txt_data": f"/dataset/mixed_text_seed{SEED}_768.jsonl",
        "vis_json": f"/dataset/mixed_calibration_seed{SEED}_192.json",
        "vis_imgs": "/dataset",
        "save_to": f"./llava-v1.6-mistral-7b-cma-50-seed{SEED}",
        "stats_cache": f"./cma_Log1p_LLaVA_v1.6_mistral_stats_7b_seed{SEED}.pt" 
    }

    total_start = time.time()

    pruner = LlavaCmaPruner(
        CONFIG["model"], 
        sparsity=0.50, 
        omega=1.0, 
        tau_0=1.0, 
        gamma=1.0
    )
    pruner.load_model()
    
    t_load = time.time() - total_start
    print(f"⏱️ Time Taken to Load the Model: {t_load:.2f} second\n")

    t_calib_start = time.time()
    if not Path(CONFIG["stats_cache"]).exists():
        pruner.run_calibration(CONFIG["txt_data"], CONFIG["vis_json"], CONFIG["vis_imgs"], num=None)
        pruner.save_calibration_stats(CONFIG["stats_cache"])
    else:
        pruner.load_calibration_stats(CONFIG["stats_cache"])
    t_calib = time.time() - t_calib_start
    print(f"⏱️ Time required for feature extraction: {t_calib:.2f} second\n")

    t_prune_start = time.time()
    pruner.execute_pruning()
    t_prune = time.time() - t_prune_start
    print(f"⏱️ Time Taken for Core Pruning: {t_prune:.2f} second\n")

    t_save_start = time.time()
    pruner.save(CONFIG["save_to"])
    t_save = time.time() - t_save_start
    print(f"⏱️ Time Taken to Save the Model: {t_save:.2f} second\n")

    print(f"🏁 The process is now complete! Total time:: {time.time() - total_start:.2f} second")

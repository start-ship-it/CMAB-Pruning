"""
CMA_Log1P 剪枝算法 - 完美适配 LLaVA-v1.6-Mistral-7B-HF
特性：
1. 适配 LLaVA-v1.6-Mistral 的层级结构与 LlavaNext 架构。
2. 自动切换为 Mistral 专属的 [INST] 模板进行多模态校准。
3. 支持全量数据加载 (num=None) 与完整的激活值缓存机制。
4. 保存时自动注入对齐 Mistral 格式的 Chat Template，彻底清除兼容性隐患。
5. 🌟 纯净版信息论驱动的跨模态动态自适应平衡因子 (IT-CMG)，无边界约束。
"""

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

# ==========================================
# 1. 跨模态校准集构建
# ==========================================
class MultimodalCalibration:
    @staticmethod
    def get_text_samples(jsonl_path, num_samples=None):
        print(f"📚 正在加载文本集: {jsonl_path}")
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
        except Exception as e: print(f"❌ 读取失败: {e}")
        print(f"✅ 成功加载 {len(texts)} 条文本样本。")
        return texts

    @staticmethod
    def get_vision_samples(json_path, image_folder, num_samples=None):
        print(f"🖼️ 正在加载图文指令集: {json_path}")
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
        print(f"✅ 成功加载 {len(samples)} 条图文样本。")
        return samples

# ==========================================
# 2. CMA 激活收集器
# ==========================================
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
        raise AttributeError("无法定位 Transformer 层，请检查模型架构。")

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
        """🌟 纯净版信息论驱动的动态跨模态自适应门控机制 (IT-CMG)"""
        s = self.act_stats[name]
        eps = 1e-8
        
        a_t = torch.sqrt(s['t_sq'] / s['t_cnt']) if s['t_cnt'] > 0 else torch.zeros_like(s['t_sq'])
        a_v = torch.sqrt(s['v_sq'] / s['v_cnt']) if s['v_cnt'] > 0 else torch.zeros_like(s['v_sq'])
        a_t, a_v = a_t.float(), a_v.float()
        
        # 对数平滑 (Log1p)
        log_t = torch.log1p(a_t)
        log_v = torch.log1p(a_v)
        
        # 无穷范数归一化
        norm_t = log_t / (torch.max(log_t) + eps)
        norm_v = log_v / (torch.max(log_v) + eps)

        # 概率空间分布
        p_t = norm_t / (torch.sum(norm_t) + eps)
        p_v = norm_v / (torch.sum(norm_v) + eps)
        
        # 能量项 E_l
        energy_t = torch.sum(norm_t.pow(2))
        energy_v = torch.sum(norm_v.pow(2))
        E_l = torch.log((energy_v + eps) / (energy_t + eps))
        
        # 差异特异性方差对比度
        var_t = torch.var(norm_t)
        var_v = torch.var(norm_v)
        delta_sigma_l = (var_v - var_t) / (var_v + var_t + eps)
        
        # 詹森-香农散度 JS Div
        M = 0.5 * (p_v + p_t)
        kl_v = torch.sum(p_v * torch.log((p_v + eps) / (M + eps)))
        kl_t = torch.sum(p_t * torch.log((p_t + eps) / (M + eps)))
        js_div = 0.5 * kl_v + 0.5 * kl_t
        
        # 纯净态路由对数几率 G_l
        G_l = E_l + omega * delta_sigma_l * js_div
        
        # 稀疏度调制阀
        tau = tau_0 * ((1.0 - sparsity) ** gamma)
        tau = max(tau, 1e-4) 
        
        # 最终自适应平衡因子 lambda_l
        lambda_l = 1.0 / (1.0 + torch.exp(-G_l / tau))

        # 跨模态融合
        act_total = lambda_l * norm_v + (1.0 - lambda_l) * norm_t
        
        return act_total, lambda_l.item()

# ==========================================
# 3. 剪枝执行引擎
# ==========================================
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
        print(f"📦 正在载入 LLaVA-v1.6-Mistral-7B: {self.model_path}")
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
        print(f"💾 保存统计量至: {save_path}")
        torch.save(self.profiler.act_stats, save_path)

    def load_calibration_stats(self, load_path):
        print(f"🔄 从缓存加载统计量: {load_path}")
        if self.profiler is None: self.profiler = CMAProfiler(self.model)
        self.profiler.act_stats = torch.load(load_path, map_location="cpu", weights_only=False)

    def execute_pruning(self):
        print(f"✂️ 执行 IT-CMG 纯净版自适应剪枝 (Sparsity={self.sparsity})...")
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

        print("\n📊 ==================== IT-CMG 动态平衡路由最终数值报告 ====================")
        if self.layer_lambdas_avg:
            print(f"📈 全局网络平均视觉倾向度 (Global Mean Lambda): {np.mean(self.layer_lambdas_avg):.4f}")
            if attn_lambdas: print(f"🧩 自注意力机制模块平均倾向度 (Attn Blocks Mean Lambda): {np.mean(attn_lambdas):.4f}")
            if mlp_lambdas: print(f"🧬 前馈网络结构模块平均倾向度 (MLP Blocks Mean Lambda): {np.mean(mlp_lambdas):.4f}")
            print("-" * 76)
            mid_idx = len(self.layer_lambdas_avg) // 2
            print(f"  - 前半段浅层网络 (Layers 0-{mid_idx-1}) 平均 Lambda: {np.mean(self.layer_lambdas_avg[:mid_idx]):.4f}")
            print(f"  - 后半段深层网络 (Layers {mid_idx}-{len(self.layer_lambdas_avg)-1}) 平均 Lambda: {np.mean(self.layer_lambdas_avg[mid_idx:]):.4f}")
        print("===========================================================================\n")

    def save(self, path):
        print(f"💾 保存剪枝模型至: {path}")
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
            
        print("🎉 模型保存及自适应统计量保存全部完成！")

# ==========================================
# 4. 运行配置
# ==========================================
if __name__ == "__main__":
    CONFIG = {
        "model": "/root/autodl-tmp/llava-hf/llava-v1.6-mistral-7b-hf", 
        "txt_data": "/root/autodl-tmp/CMA-Pruning/data/text_only/text_256_3.jsonl",
        "vis_json": "/root/autodl-tmp/CMA-Pruning/data/image_text/merged_vision_32_6.json",
        "vis_imgs": "/root/autodl-tmp/CMA-Pruning/data/image_text",
        "save_to": "./llava-v1.6-mistral-7b-cma-50",
        "stats_cache": "./cma_Log1p_LLaVA_v1.6_mistral_stats_7b.pt" 
    }

    total_start = time.time()

    # 🌟 恢复初始纯净版超参数：omega=1.0, tau_0=1.0
    pruner = LlavaCmaPruner(
        CONFIG["model"], 
        sparsity=0.65, 
        omega=1.0, 
        tau_0=1.0, 
        gamma=1
    )
    pruner.load_model()
    
    t_load = time.time() - total_start
    print(f"⏱️ 模型加载耗时: {t_load:.2f} 秒\n")

    t_calib_start = time.time()
    if not Path(CONFIG["stats_cache"]).exists():
        pruner.run_calibration(CONFIG["txt_data"], CONFIG["vis_json"], CONFIG["vis_imgs"], num=None)
        pruner.save_calibration_stats(CONFIG["stats_cache"])
    else:
        pruner.load_calibration_stats(CONFIG["stats_cache"])
    t_calib = time.time() - t_calib_start
    print(f"⏱️ 特征提取耗时: {t_calib:.2f} 秒\n")

    t_prune_start = time.time()
    pruner.execute_pruning()
    t_prune = time.time() - t_prune_start
    print(f"⏱️ 核心剪枝耗时: {t_prune:.2f} 秒\n")

    t_save_start = time.time()
    pruner.save(CONFIG["save_to"])
    t_save = time.time() - t_save_start
    print(f"⏱️ 模型保存耗时: {t_save:.2f} 秒\n")

    print(f"🏁 流程全部结束！总共耗时: {time.time() - total_start:.2f} 秒")
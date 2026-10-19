"""4bit 模型 / LoRA 加载工具。"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        print("本地 tokenizer 未找到,尝试联网加载...")
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_model_and_tokenizer(
    model_name: str,
    adapter: Optional[str] = None,
    for_training: bool = False,
    precision: str = "auto",
    device_map: str = "cuda",
) -> Tuple[object, object, bool]:
    """加载 4bit base + 可选 LoRA adapter。

    返回 (model, tokenizer, is_qwen3)。
    """
    try:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_qwen3 = (
        str(getattr(config, "model_type", "")).lower() == "qwen3"
        or "qwen3" in model_name.lower()
    )
    use_bf16 = precision == "bf16" or (precision == "auto" and is_qwen3)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    resolved_map = {"": 0} if device_map == "cuda" else device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=resolved_map,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        print("本地模型未找到,尝试联网加载...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=resolved_map,
            trust_remote_code=True,
        )

    tokenizer = load_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter:
        if not os.path.isdir(adapter):
            raise FileNotFoundError(f"adapter not found: {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=for_training)

    if for_training:
        # 从已有 LoRA adapter 继续训练时,prepare_model_for_kbit_training 作用在
        # PeftModel 上容易把 adapter 状态弄乱;只在从 base 开始 SFT 时调用。
        if adapter is None:
            model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    return model, tokenizer, is_qwen3


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


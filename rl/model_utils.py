"""公共加载工具:本地缓存优先,避免每次联网超时。

⚠️ 现状说明(2026-10-12 审计):本模块目前**没有被任何脚本 import**,
属于预留的工具函数(各训练脚本各自保留了自己的加载逻辑)。
保留原因:统一加载入口的候选实现;如需启用,请改各脚本为 `from model_utils import ...`。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name):
    try:
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        print("本地缓存未找到,尝试联网加载(网络受限时可设置 HF_ENDPOINT 镜像)...")
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_model(model_name, bnb_config):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        print("本地缓存未找到,尝试联网加载...")
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )

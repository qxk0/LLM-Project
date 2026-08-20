"""公共加载工具:本地缓存优先,避免每次联网超时。"""

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

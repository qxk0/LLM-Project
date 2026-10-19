# 顶层文件 (requirements / 启动脚本)

> 生成时间：2026-10-14 17:27:04

## 目录结构

```text

```

## 文件内容

### `requirements.txt`

```text
# HuggingFace 工具链(已在 .venv 中装好)
# PyTorch 2.5.1+cu121 是从官方 CUDA 索引单独安装的,见 README
transformers
datasets
accelerate
peft
trl
bitsandbytes
sentencepiece
tiktoken
matplotlib
```

### `start.ps1`

```powershell
# 一键进入项目环境:激活虚拟环境 + 把所有缓存指到 D 盘
# 用法:每次新开终端后运行 .\start.ps1
$env:PIP_CACHE_DIR = 'D:\LLM-Project\.cache\pip'
$env:HF_HOME = 'D:\LLM-Project\.cache\huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
& D:\LLM-Project\.venv\Scripts\Activate.ps1
```

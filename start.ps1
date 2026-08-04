# 一键进入项目环境:激活虚拟环境 + 把所有缓存指到 D 盘
# 用法:每次新开终端后运行 .\start.ps1
$env:PIP_CACHE_DIR = 'D:\LLM-Project\.cache\pip'
$env:HF_HOME = 'D:\LLM-Project\.cache\huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
& D:\LLM-Project\.venv\Scripts\Activate.ps1

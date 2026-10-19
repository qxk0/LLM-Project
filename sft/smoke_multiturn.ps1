param(
    [string]$ModelName = "Qwen/Qwen3-0.6B",
    [string]$DataDir = "data\multiturn_smoke",
    [string]$OutputDir = "models\sft_multiturn_smoke"
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$builder = Join-Path $PSScriptRoot "..\data_eng\build_multi_turn.py"
$trainer = Join-Path $PSScriptRoot "train_multiturn_sft.py"

& $python $builder --smoke-test --out-dir $DataDir
if ($LASTEXITCODE -ne 0) {
    throw "multi-turn data build failed with exit code $LASTEXITCODE"
}

$sftData = Join-Path $DataDir "sft_train.jsonl"
& $python $trainer `
    --smoke-test `
    --model-name $ModelName `
    --data-file $sftData `
    --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "multi-turn SFT failed with exit code $LASTEXITCODE"
}

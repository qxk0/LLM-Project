param(
    [string]$ModelName = "Qwen/Qwen3-0.6B",
    [string]$DataFile = "runs\2026-10-12\data\clean\train.jsonl",
    [string]$OutputDir = "models\sft_qwen3_smoke"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $DataFile)) {
    throw "Data file not found: $DataFile"
}

# Windows/WDDM does not support expandable_segments; keep the default allocator.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainScript = Join-Path $PSScriptRoot "train_sft.py"

& $python $trainScript `
    --model-name $ModelName `
    --data-file $DataFile `
    --smoke-test `
    --max-examples 32 `
    --max-steps 10 `
    --max-seq-len 256 `
    --batch-size 1 `
    --grad-accum 4 `
    --lora-r 8 `
    --lora-alpha 16 `
    --lr 1e-4 `
    --device-map cuda `
    --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed with exit code $LASTEXITCODE"
}

param(
    [int]$MaxPrompts = 400,
    [int]$DpoSteps = 100,
    [int]$SimpoSteps = 100,
    [int]$GrpoSteps = 100
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$pairs = "runs\2026-10-12\alignment\pairs.jsonl"

& $python rl\alignment\build_preferences.py `
    --max-prompts $MaxPrompts --num-samples 4 `
    --output $pairs
if ($LASTEXITCODE -ne 0) { throw "preference build failed" }

& $python rl\train_dpo_alignment.py `
    --pairs $pairs --max-steps $DpoSteps --output-dir models\alignment_dpo
if ($LASTEXITCODE -ne 0) { throw "DPO failed" }

& $python rl\train_simpo_alignment.py `
    --pairs $pairs --max-steps $SimpoSteps --output-dir models\alignment_simpo
if ($LASTEXITCODE -ne 0) { throw "SimPO failed" }

& $python rl\train_grpo_alignment.py `
    --max-steps $GrpoSteps --output-dir models\alignment_grpo
if ($LASTEXITCODE -ne 0) { throw "GRPO failed" }

& $python sft\eval_domain.py `
    --data-file runs\2026-10-12\data\clean\test.jsonl `
    --prompt-format chat --max-new-tokens 64 `
    --adapters runs\2026-10-12\models\sft models\alignment_dpo models\alignment_simpo models\alignment_grpo `
    --predictions-dir runs\alignment_compare\predictions `
    --report runs\alignment_compare\eval_chat.md
if ($LASTEXITCODE -ne 0) { throw "evaluation failed" }

Write-Host "Alignment suite finished: runs/alignment_compare/eval_chat.md"

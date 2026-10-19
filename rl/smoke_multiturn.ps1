param(
    [string]$DataDir = "data\multiturn_smoke",
    [string]$SftAdapter = "models\sft_multiturn_smoke"
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

if (-not (Test-Path -LiteralPath $SftAdapter)) {
    throw "SFT adapter not found: $SftAdapter. Run sft/smoke_multiturn.ps1 first."
}

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_multiturn_grpo.py"
$evaler = Join-Path $PSScriptRoot "eval_multiturn.py"
$trainEpisodes = Join-Path $DataDir "episodes_train.jsonl"
$testEpisodes = Join-Path $DataDir "episodes_test.jsonl"

& $python $trainer `
    --smoke-test `
    --reward-mode turn `
    --max-steps 1 `
    --max-episodes 2 `
    --num-generations 2 `
    --adapter $SftAdapter `
    --data-file $trainEpisodes `
    --output-dir "models\multiturn_grpo_turn_smoke"

& $python $trainer `
    --smoke-test `
    --reward-mode terminal `
    --max-steps 1 `
    --max-episodes 2 `
    --num-generations 2 `
    --adapter $SftAdapter `
    --data-file $trainEpisodes `
    --output-dir "models\multiturn_grpo_terminal_smoke"

& $python $evaler `
    --teacher `
    --smoke-test `
    --data-file $testEpisodes `
    --output-dir "runs\multiturn_smoke\eval_teacher"

Write-Host "Multi-turn smoke finished."

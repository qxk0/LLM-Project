param(
    [string]$DataDir = "data\multiturn",
    [string]$SftAdapter = "models\sft_multiturn_all_early",
    [string]$DataFile = "",
    [string]$Curriculum = "8",
    [int]$Steps = 10,
    [int]$Generations = 4,
    [int]$PerHorizonLimit = 0
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

if (-not (Test-Path -LiteralPath $SftAdapter)) {
    throw "SFT adapter not found: $SftAdapter. Train it with sft/train_multiturn_sft.py first."
}

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_multiturn_grpo.py"
$evaler = Join-Path $PSScriptRoot "eval_multiturn.py"
if ([string]::IsNullOrWhiteSpace($DataFile)) {
    $DataFile = Join-Path $DataDir "episodes_train_all.jsonl"
}
$trainEpisodes = $DataFile
$testEpisodes = Join-Path $DataDir "episodes_test.jsonl"

& $python $trainer `
    --reward-mode terminal `
    --adapter $SftAdapter `
    --data-file $trainEpisodes `
    --curriculum $Curriculum `
    --max-steps $Steps `
    --num-generations $Generations `
    --output-dir "models\multiturn_grpo_terminal"
if ($LASTEXITCODE -ne 0) {
    throw "terminal-only training failed with exit code $LASTEXITCODE"
}

& $python $trainer `
    --reward-mode turn `
    --adapter $SftAdapter `
    --data-file $trainEpisodes `
    --curriculum $Curriculum `
    --max-steps $Steps `
    --num-generations $Generations `
    --output-dir "models\multiturn_grpo_turn"
if ($LASTEXITCODE -ne 0) {
    throw "turn-level training failed with exit code $LASTEXITCODE"
}

$evalArgs = @(
    "--adapters", $SftAdapter, "models\multiturn_grpo_terminal", "models\multiturn_grpo_turn",
    "--data-file", $testEpisodes,
    "--output-dir", "runs\multiturn_eval"
)
if ($PerHorizonLimit -gt 0) {
    $evalArgs += @("--per-horizon-limit", "$PerHorizonLimit")
}
& $python $evaler @evalArgs
if ($LASTEXITCODE -ne 0) {
    throw "multi-turn evaluation failed with exit code $LASTEXITCODE"
}

Write-Host "Ablation finished: runs/multiturn_eval/report.md"

param(
    [string]$Model = "gpt-4.1-mini",
    [ValidateSet("openai_compatible", "openai_responses", "anthropic", "gemini")]
    [string]$Provider = "openai_compatible",
    [string]$EnvFile = ".env.closed",
    [string]$OutputDir = "outputs_daic_closed_eval",
    [string]$Profiles = "data/daic/patient_profiles/daic_dialogue_derived_patient_profiles.jsonl",
    [string]$Schema = "schemas/daic_symptom_slot_schema.json",
    [string]$GroupDir = "data/daic/profile_split",
    [string]$CanonicalDir = "data/daic/canonical_evidence",
    [string]$Splits = "test",
    [int]$MaxProfiles = 219,
    [int]$MaxGroups = 10000,
    [int]$MaxPerSlot = 999,
    [int]$MaxTurns = 24,
    [int]$MaxIterations = 32,
    [int]$Limit = 0,
    [int]$MaxOutputTokens = 96,
    [double]$Temperature = 0.0,
    [string]$Python = "",
    [switch]$SkipMetrics
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $Python = if ($env:AR_GRPO_PYTHON) { $env:AR_GRPO_PYTHON } else { "python" }
}

function Count-Jsonl {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    $count = 0
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    foreach ($line in [System.IO.File]::ReadLines($resolved)) {
        if ($line.Trim().Length -gt 0) {
            $count += 1
        }
    }
    return $count
}

function Require-Path {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Description at: $Path"
    }
}

Require-Path $Profiles "DAIC profile file"
Require-Path $Schema "DAIC schema file"
Require-Path $GroupDir "DAIC profile split directory"
Require-Path $CanonicalDir "DAIC canonical evidence directory"
if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Missing env file at: $EnvFile. Create it with API keys before running this script."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$safeModel = [Regex]::Replace($Model, "[^A-Za-z0-9_.-]+", "_")
$doctorOutputPath = Join-Path $OutputDir "$($Provider)_$($safeModel)_doctor_outputs.jsonl"
$pendingPath = Join-Path $OutputDir "daic_llm_doctor_online_replay_pending_requests.jsonl"
$recordsPath = Join-Path $OutputDir "daic_llm_doctor_online_replay_records.jsonl"
$analysisDir = Join-Path $OutputDir "tree_aligned_canonical_recovery"

Write-Host "DAIC closed-model evaluation"
Write-Host "  provider: $Provider"
Write-Host "  model:    $Model"
Write-Host "  output:   $OutputDir"
Write-Host "  cache:    $doctorOutputPath"

for ($iteration = 1; $iteration -le $MaxIterations; $iteration++) {
    Write-Host ""
    Write-Host "== Iteration ${iteration}: replay =="

    & $Python scripts/run_llm_doctor_online_replay.py `
        --profiles $Profiles `
        --schema $Schema `
        --group-dir $GroupDir `
        --dataset-prefix daic `
        --splits $Splits `
        --language en `
        --patient-controller-version v3_2 `
        --provider cached `
        --model-output-path $doctorOutputPath `
        --missing-output-policy stop `
        --max-profiles $MaxProfiles `
        --max-groups $MaxGroups `
        --max-per-slot $MaxPerSlot `
        --max-turns $MaxTurns `
        --output-dir $OutputDir

    $pendingCount = Count-Jsonl $pendingPath
    Write-Host "pending requests: $pendingCount"
    if ($pendingCount -eq 0) {
        Write-Host "No pending requests. Replay is complete."
        break
    }

    Write-Host "== Iteration ${iteration}: call closed model =="
    & $Python scripts/call_closed_llm_for_pending_requests.py `
        --provider $Provider `
        --model $Model `
        --env-file $EnvFile `
        --pending-path $pendingPath `
        --output-path $doctorOutputPath `
        --limit $Limit `
        --max-output-tokens $MaxOutputTokens `
        --temperature $Temperature

    if ($iteration -eq $MaxIterations) {
        throw "Reached MaxIterations=$MaxIterations with pending requests still remaining."
    }
}

if (-not $SkipMetrics) {
    Write-Host ""
    Write-Host "== Analyze canonical evidence recovery =="
    & $Python scripts/analyze_tree_aligned_canonical_evidence_recovery.py `
        --canonical-dir $CanonicalDir `
        --dataset-prefix daic `
        --records "$safeModel=$recordsPath" `
        --output-dir $analysisDir
}

Write-Host ""
Write-Host "Done."
Write-Host "Replay records: $recordsPath"
Write-Host "Doctor outputs:  $doctorOutputPath"
if (-not $SkipMetrics) {
    Write-Host "Metrics:         $(Join-Path $analysisDir 'tree_aligned_canonical_evidence_recovery_summary.json')"
}

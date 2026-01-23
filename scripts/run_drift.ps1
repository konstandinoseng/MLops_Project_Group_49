# Run drift detection job (PowerShell)
# Usage: .\scripts\run_drift.ps1

$ErrorActionPreference = "Stop"

# Configuration (override via environment or edit defaults)
if (-not $env:GCS_BUCKET) { $env:GCS_BUCKET = "project-094ec169-6184-46b3-838_cloudbuild" }
if (-not $env:GCS_PREFIX) { $env:GCS_PREFIX = "logging" }
if (-not $env:REFERENCE_DATA_PATH) { $env:REFERENCE_DATA_PATH = "data/raw/Sentences_AllAgree.txt" }
if (-not $env:EMBEDDING_MODEL) { $env:EMBEDDING_MODEL = "all-MiniLM-L6-v2" }

Write-Host "=== Running Drift Detection ==="
Write-Host "GCS_BUCKET: $env:GCS_BUCKET"
Write-Host "GCS_PREFIX: $env:GCS_PREFIX"

# Run the drift detection script
uv run python -m project.monitoring.drift

Write-Host "=== Done ==="

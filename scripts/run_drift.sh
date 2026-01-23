#!/bin/bash
# Run drift detection job
# Usage: ./scripts/run_drift.sh

set -e

# Configuration (override via environment or edit defaults)
export GCS_BUCKET="${GCS_BUCKET:-project-094ec169-6184-46b3-838_cloudbuild}"
export GCS_PREFIX="${GCS_PREFIX:-logging}"
export REFERENCE_DATA_PATH="${REFERENCE_DATA_PATH:-data/raw/Sentences_AllAgree.txt}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"

echo "=== Running Drift Detection ==="
echo "GCS_BUCKET: $GCS_BUCKET"
echo "GCS_PREFIX: $GCS_PREFIX"

# Run the drift detection script
python -m project.monitoring.drift

echo "=== Done ==="

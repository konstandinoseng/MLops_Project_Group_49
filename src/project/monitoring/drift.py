"""
Drift detection batch job.

Reads inference logs from GCS, builds embeddings, compares against
reference data using Evidently, and writes a report back to GCS.

Usage:
    GCS_BUCKET=... GCS_PREFIX=logging uv run python -m project.monitoring.drift
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from google.cloud import storage
except ImportError:
    storage = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment, misc]

try:
    from evidently import ColumnMapping
    from evidently import Report
    from evidently.metrics import EmbeddingsDriftMetric
except ImportError:
    Report = None  # type: ignore[assignment, misc]
    ColumnMapping = None  # type: ignore[assignment, misc]
    EmbeddingsDriftMetric = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------
GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_PREFIX = os.getenv("GCS_PREFIX", "logging")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/raw/Sentences_AllAgree.txt")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------
def _gcs_client() -> "storage.Client":
    if storage is None:
        raise RuntimeError("google-cloud-storage is required")
    return storage.Client()


def fetch_inference_logs(limit: Optional[int] = None) -> List[dict]:
    """Fetch JSON inference logs from GCS."""
    client = _gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    prefix = f"{GCS_PREFIX}/inference/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    # Sort by name (timestamp) descending
    blobs = sorted(blobs, key=lambda b: b.name, reverse=True)
    if limit:
        blobs = blobs[:limit]

    logs = []
    for blob in blobs:
        content = blob.download_as_text()
        logs.append(json.loads(content))
    return logs


def save_report_to_gcs(html_content: str, report_name: str) -> str:
    """Save HTML report to GCS and return the object path."""
    client = _gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    object_name = f"{GCS_PREFIX}/drift-reports/{timestamp}_{report_name}.html"
    blob = bucket.blob(object_name)
    blob.upload_from_string(html_content, content_type="text/html")
    return f"gs://{GCS_BUCKET}/{object_name}"


# ---------------------------------------------------------------------------
# Reference data loader
# ---------------------------------------------------------------------------
def load_reference_texts(path: str = REFERENCE_DATA_PATH) -> List[str]:
    """Load reference texts from Financial Phrasebank format."""
    texts = []
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Format: "sentence@label"
                    if "@" in line:
                        text = line.rsplit("@", 1)[0]
                    else:
                        text = line
                    texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def build_embeddings(texts: List[str], model_name: str = EMBEDDING_MODEL) -> pd.DataFrame:
    """Convert texts to embeddings using SentenceTransformer."""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required: pip install sentence-transformers")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    df = pd.DataFrame(embeddings)
    df.columns = [f"emb_{i}" for i in range(df.shape[1])]
    return df


# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------
def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    subset_cols: int = 50,
) -> "Report":
    """Run Evidently embeddings drift report."""
    if Report is None or ColumnMapping is None or EmbeddingsDriftMetric is None:
        raise RuntimeError("evidently is required: pip install evidently")

    # Use a subset of embedding columns
    cols = list(reference_df.columns[:subset_cols])
    column_mapping = ColumnMapping(embeddings={"text_embeddings": cols})

    report = Report(metrics=[EmbeddingsDriftMetric("text_embeddings")])
    report.run(
        reference_data=reference_df[cols],
        current_data=current_df[cols],
        column_mapping=column_mapping,
    )
    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(limit: int = 100, save_to_gcs: bool = True) -> None:
    """Run drift detection pipeline."""
    print("=== Drift Detection Job ===")

    # 1. Fetch current logs
    print(f"Fetching up to {limit} inference logs from GCS...")
    logs = fetch_inference_logs(limit=limit)
    if not logs:
        print("No inference logs found. Exiting.")
        return

    current_texts = [log.get("text") for log in logs if log.get("text")]
    print(f"Found {len(current_texts)} texts in logs.")

    if len(current_texts) < 5:
        print("Not enough texts for drift detection. Need at least 5.")
        return

    # 2. Load reference data
    print(f"Loading reference data from {REFERENCE_DATA_PATH}...")
    reference_texts = load_reference_texts()
    if not reference_texts:
        print("No reference data found. Exiting.")
        return
    print(f"Loaded {len(reference_texts)} reference texts.")

    # 3. Build embeddings
    print("Building embeddings for reference data...")
    reference_df = build_embeddings(reference_texts[:500])  # Limit for speed

    print("Building embeddings for current data...")
    current_df = build_embeddings(current_texts)

    # 4. Run drift report
    print("Running drift report...")
    report = run_drift_report(reference_df, current_df)

    # 5. Save report
    if save_to_gcs and GCS_BUCKET:
        html = report.get_html()
        gcs_path = save_report_to_gcs(html, "embeddings_drift")
        print(f"Report saved to: {gcs_path}")
    else:
        # Save locally
        report.save_html("drift_report.html")
        print("Report saved to: drift_report.html")

    print("=== Done ===")


if __name__ == "__main__":
    main()

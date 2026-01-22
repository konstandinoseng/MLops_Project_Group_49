#!/bin/bash
set -e

# Configuration
GHCR_REPO="ghcr.io/konstandinoseng/mlops_project_group_49"
GCP_REGISTRY="europe-north2-docker.pkg.dev/project-094ec169-6184-46b3-838/mlops-repos"

echo "=== Authenticating to GHCR ==="
# Requires GITHUB_TOKEN env var or will prompt for password
# export GITHUB_TOKEN=your_token
echo "${GITHUB_TOKEN}" | docker login ghcr.io -u USERNAME --password-stdin || \
  echo "Note: If repo is public, GHCR auth may not be needed"

echo "=== Authenticating to GCP Artifact Registry ==="
gcloud auth configure-docker europe-north2-docker.pkg.dev --quiet

echo "=== Pulling images from GHCR ==="
docker pull ${GHCR_REPO}/gcp_test_app:latest
docker pull ${GHCR_REPO}/streamlit-frontend:latest

echo "=== Tagging images for GCP ==="
docker tag ${GHCR_REPO}/gcp_test_app:latest ${GCP_REGISTRY}/gcp_test_app:latest
docker tag ${GHCR_REPO}/streamlit-frontend:latest ${GCP_REGISTRY}/streamlit-frontend:latest

echo "=== Pushing images to GCP Artifact Registry ==="
docker push ${GCP_REGISTRY}/gcp_test_app:latest
docker push ${GCP_REGISTRY}/streamlit-frontend:latest

echo "=== Done! Images pushed to GCP ==="
echo "Backend: ${GCP_REGISTRY}/gcp_test_app:latest"
echo "Frontend: ${GCP_REGISTRY}/streamlit-frontend:latest"

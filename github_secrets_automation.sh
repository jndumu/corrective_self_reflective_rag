#!/bin/bash
export $(grep -v '^#' .env | xargs)
# This script automates the process of adding environment secrets to a GitHub repository from local environment variables.
# Prerequisites: GitHub CLI (gh) must be installed and authenticated.

# Define the repository (update this to your repository name)
REPO="jndumu/corrective_self_reflective_rag"

# Define the secrets to add (update these with your actual environment variable names)
SECRETS=(
  "OPENAI_API_KEY"
  "TAVILY_API_KEY"
  "QDRANT_URL"
  "QDRANT_API_KEY"
  "RERANKER_BACKEND"
  "UPLOAD_DIR"
)

# Loop through each secret and add it to the GitHub repository
for SECRET in "${SECRETS[@]}"; do
  VALUE=$(printenv "$SECRET")
  if [ -z "$VALUE" ]; then
    echo "[WARNING] $SECRET is not set in the local environment. Skipping..."
  else
    echo "Adding $SECRET to GitHub repository $REPO..."
    gh secret set "$SECRET" --repo "$REPO" --body "$VALUE"
    if [ $? -eq 0 ]; then
      echo "[SUCCESS] $SECRET added successfully."
    else
      echo "[ERROR] Failed to add $SECRET."
    fi
  fi
done

echo "All secrets processed."
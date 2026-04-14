#!/bin/bash
set -e

# =============================================================================
# Quick Deploy Script for GCP Cloud Run for Interview with Persistent Storage
# =============================================================================

# TODO: CONFIGURATION (fill this)
PROJECT_ID="llm-interviewer-491820"
SERVICE_NAME="sparkme-web-interview"
REGION="us-central1"
BUCKET_NAME="sparkme-web-interview-${PROJECT_ID}"

# We build a comma-separated string of env vars to pass to Cloud Run
NON_SENSITIVE_VARS=""
NON_SENSITIVE_VARS+="MODEL_NAME=gpt-5.1,"
NON_SENSITIVE_VARS+="SESSION_SCRIBE_MODEL_NAME=gpt-5.1,"
NON_SENSITIVE_VARS+="STRATEGIC_PLANNER_MODEL_NAME=gpt-5.1,"
NON_SENSITIVE_VARS+="EMBEDDING_BACKEND=openai,"
NON_SENSITIVE_VARS+="MAX_EVENTS_LEN=20,"
NON_SENSITIVE_VARS+="MAX_CONSIDERATION_ITERATIONS=10,"
NON_SENSITIVE_VARS+="USE_BASELINE_PROMPT=false,"
NON_SENSITIVE_VARS+="EVAL_MODE=false,"
NON_SENSITIVE_VARS+="COMPLETION_METRIC=minimum_threshold,"

# Directories (Pointing to the Persistent Mount /app/data)
NON_SENSITIVE_VARS+="LOGS_DIR=/app/data/logs,"
NON_SENSITIVE_VARS+="DATA_DIR=/app/data/data,"
NON_SENSITIVE_VARS+="INTERVIEW_PLAN_PATH_INTAKE=/app/configs/topics_intake.json,"
NON_SENSITIVE_VARS+="INTERVIEW_PLAN_PATH_WEEKLY=/app/configs/topics_weekly.json,"
NON_SENSITIVE_VARS+="USER_PORTRAIT_PATH=/app/configs/user_portrait.json,"

# Strategic planner config
NON_SENSITIVE_VARS+="STRATEGIC_PLANNER_TURN_TRIGGER=3,"
NON_SENSITIVE_VARS+="STRATEGIC_PLANNER_NUM_ROLLOUTS=3,"
NON_SENSITIVE_VARS+="STRATEGIC_PLANNER_ROLLOUT_HORIZON=3,"
NON_SENSITIVE_VARS+="STRATEGIC_PLANNER_MAX_QUESTIONS=5,"
NON_SENSITIVE_VARS+="STRATEGIC_PLANNER_ALPHA=1,"
NON_SENSITIVE_VARS+="STRATEGIC_PLANNER_BETA=1,"
NON_SENSITIVE_VARS+="STRATEGIC_PLANNER_GAMMA=0.5,"
NON_SENSITIVE_VARS+="STRATEGIC_PLANNER_MIN_NOVELTY=3,"

# Session Defaults
NON_SENSITIVE_VARS+="SESSION_TIMEOUT_MINUTES=30,"
NON_SENSITIVE_VARS+="MEMORY_THRESHOLD_FOR_UPDATE=15"

echo "================================================"
echo "Deploying to GCP Cloud Run with Cloud Storage"
echo "================================================"
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"
echo "================================================"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling APIs..."
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com

# Create Cloud Storage bucket for persistent data
echo "Creating Cloud Storage bucket..."
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME 2>/dev/null || echo "Bucket already exists"

# 3. SYNC CONFIGS (The Fix for 'Mount Masking')
# We upload your local ./data files to the bucket BEFORE deploying.
# This ensures the app can see them when the bucket is mounted.
echo "📂 Syncing local config files to Bucket..."
# gcloud storage cp -r ./data/sample_user_profiles gs://$BUCKET_NAME/
gcloud storage cp -r ./configs gs://$BUCKET_NAME/

# Build the container image using the Dockerfile (pins Python 3.10)
IMAGE="us-central1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$SERVICE_NAME:latest"

echo "Building container image..."
gcloud builds submit . \
  --region $REGION \
  --config scripts/web_interview/cloudbuild.yaml \
  --substitutions "_IMAGE=$IMAGE" \
  --gcs-log-dir "gs://$BUCKET_NAME/build-logs"

# Deploy to Cloud Run with mounted storage (Gen2)
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "$NON_SENSITIVE_VARS" \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest" \
  --set-secrets "FLASK_SECRET_KEY=flask-secret-key:latest" \
  --add-volume name=data,type=cloud-storage,bucket=$BUCKET_NAME \
  --add-volume-mount volume=data,mount-path=/app/data

# Get the URL
echo ""
echo "================================================"
echo "Deployment complete!"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
echo ""
echo "Share this link with users:"
echo "   $SERVICE_URL/login"
echo ""
echo "Health check:"
echo "   $SERVICE_URL/health"
echo ""
echo "Data stored in:"
echo "   gs://$BUCKET_NAME"
echo ""
echo "================================================"
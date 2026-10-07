#!/usr/bin/env bash
# ==============================================================================
# Telecom Customer Churn Prediction & Retention System - AWS Deployment Automation
# Automates ECR authentication, container building, tagging, pushing, and ECS update.
# ==============================================================================

set -euo pipefail

# Configuration Defaults (override via environment variables if desired)
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-123456789012}"
ECR_API_REPO="telecom-churn-api"
ECR_UI_REPO="telecom-churn-ui"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ECS_CLUSTER="${ECS_CLUSTER:-telecom-churn-cluster}"
ECS_SERVICE="${ECS_SERVICE:-telecom-churn-service}"

echo "========================================================================"
echo " Starting AWS Deployment Workflow for Telecom Churn Prediction System "
echo " Region:        $AWS_REGION"
echo " Account ID:    $AWS_ACCOUNT_ID"
echo " Tag:           $IMAGE_TAG"
echo "========================================================================"

# Step 1: Authenticate Docker with AWS ECR
echo "[1/5] Authenticating Docker with Amazon ECR in $AWS_REGION..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Step 2: Ensure ECR repositories exist
echo "[2/5] Ensuring ECR repositories exist..."
aws ecr describe-repositories --repository-names "$ECR_API_REPO" --region "$AWS_REGION" > /dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$ECR_API_REPO" --region "$AWS_REGION" --image-scanning-configuration scanOnPush=true

aws ecr describe-repositories --repository-names "$ECR_UI_REPO" --region "$AWS_REGION" > /dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$ECR_UI_REPO" --region "$AWS_REGION" --image-scanning-configuration scanOnPush=true

# Step 3: Build Docker images locally
echo "[3/5] Building Docker images..."
ECR_API_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_API_REPO}:${IMAGE_TAG}"
ECR_UI_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_UI_REPO}:${IMAGE_TAG}"

docker build -f Dockerfile.api -t "$ECR_API_URI" .
docker build -f Dockerfile.ui -t "$ECR_UI_URI" .

# Step 4: Push Docker images to Amazon ECR
echo "[4/5] Pushing images to Amazon ECR..."
docker push "$ECR_API_URI"
docker push "$ECR_UI_URI"

# Step 5: Trigger rolling deployment on Amazon ECS (if cluster exists)
echo "[5/5] Checking ECS Service and initiating rolling deployment..."
if aws ecs describe-clusters --clusters "$ECS_CLUSTER" --region "$AWS_REGION" > /dev/null 2>&1; then
    echo "Updating ECS service $ECS_SERVICE..."
    aws ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" --force-new-deployment --region "$AWS_REGION"
    echo "Deployment triggered successfully! Monitoring rollout..."
else
    echo "Notice: ECS cluster '$ECS_CLUSTER' not found or credentials not provisioned. Container images pushed to ECR successfully."
fi

echo "========================================================================"
echo " AWS Deployment Process Complete!"
echo " ECR API Image: $ECR_API_URI"
echo " ECR UI Image:  $ECR_UI_URI"
echo "========================================================================"

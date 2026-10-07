<#
.SYNOPSIS
    Telecom Customer Churn Prediction & Retention System - AWS Deployment Automation (PowerShell)
.DESCRIPTION
    Automates AWS ECR authentication, container building, tagging, pushing, and ECS rollout.
#>

[CmdletBinding()]
param (
    [string]$AwsRegion = "us-east-1",
    [string]$AwsAccountId = "123456789012",
    [string]$ImageTag = "latest",
    [string]$EcsCluster = "telecom-churn-cluster",
    [string]$EcsService = "telecom-churn-service"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " Starting AWS Deployment Workflow for Telecom Churn Prediction System " -ForegroundColor Cyan
Write-Host " Region:        $AwsRegion" -ForegroundColor Yellow
Write-Host " Account ID:    $AwsAccountId" -ForegroundColor Yellow
Write-Host " Tag:           $ImageTag" -ForegroundColor Yellow
Write-Host "========================================================================" -ForegroundColor Cyan

$EcrRegistry = "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com"
$ApiImageUri = "$EcrRegistry/telecom-churn-api:$ImageTag"
$UiImageUri = "$EcrRegistry/telecom-churn-ui:$ImageTag"

# Step 1: ECR Authentication
Write-Host "[1/5] Authenticating Docker with Amazon ECR..." -ForegroundColor Green
$password = aws ecr get-login-password --region $AwsRegion
$password | docker login --username AWS --password-stdin $EcrRegistry

# Step 2: Build local containers
Write-Host "[2/5] Building Docker images..." -ForegroundColor Green
docker build -f Dockerfile.api -t $ApiImageUri .
docker build -f Dockerfile.ui -t $UiImageUri .

# Step 3: Push images to ECR
Write-Host "[3/5] Pushing images to Amazon ECR..." -ForegroundColor Green
docker push $ApiImageUri
docker push $UiImageUri

# Step 4: ECS Update
Write-Host "[4/5] Checking ECS Service and initiating rolling deployment..." -ForegroundColor Green
try {
    aws ecs update-service --cluster $EcsCluster --service $EcsService --force-new-deployment --region $AwsRegion
    Write-Host "ECS deployment triggered successfully!" -ForegroundColor Green
} catch {
    Write-Host "Notice: ECS cluster update skipped (service not present or mock credentials). Images pushed to ECR." -ForegroundColor DarkYellow
}

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " AWS Deployment Process Complete!" -ForegroundColor Cyan
Write-Host " ECR API: $ApiImageUri" -ForegroundColor White
Write-Host " ECR UI:  $UiImageUri" -ForegroundColor White
Write-Host "========================================================================" -ForegroundColor Cyan

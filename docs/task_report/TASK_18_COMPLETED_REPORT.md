# TASK 18 — Cloud Deployment Architecture & AWS Hosting Plan
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System  
**Task ID**: TASK 18  
**Status**: COMPLETED  
**Executed On**: 2026-10-07  
**Assignment Requirement**: Requirement #15 ("Deployment - Use Docker and AWS")

---

## 1. Objective

Formulate an end-to-end cloud deployment strategy for hosting the containerized Telecom Customer Churn Prediction & Retention System on Amazon Web Services (AWS). Provide automated deployment scripts, production task definitions, reverse proxy routing rules, and comprehensive cloud architecture documentation.

---

## 2. Cloud Architecture Architecture & Deliverables

### 2.1 Artifacts Created & Configured

| File | Purpose | Key Details |
|---|---|---|
| `docs/aws_deployment_guide.md` | Comprehensive AWS Architecture Guide | Complete architecture designs, comparison of ECS Fargate vs App Runner vs EC2, step-by-step ECR push commands, Route 53, ACM SSL, CloudWatch logging, cost estimation table |
| `infra/ecs-task-definition.json` | AWS ECS Fargate Task Definition | Dual-container configuration (1024 CPU units, 2048 MB RAM), health check definitions, AWS CloudWatch logging driver, non-root user bindings |
| `infra/nginx.conf` | Production Reverse Proxy Configuration | Upstream routing for API (`/api/`, `/health`, `/docs`) and Streamlit UI (`/`) with WebSocket upgrade headers (`Upgrade: $http_upgrade`) |
| `scripts/deploy_aws.sh` | Automated Bash Deployment Script | Authenticates with ECR, creates repositories, builds local images, pushes images, and triggers ECS service rolling update |
| `scripts/deploy_aws.ps1` | Automated PowerShell Deployment Script | Native Windows/PowerShell automation for ECR authentication, tagging, pushing, and ECS rolling deployment |

### 2.2 Production Architecture Topology (AWS ECS Fargate + ALB)

```
[ Internet Users ]
       │
       ▼ (HTTPS: Port 443)
[ AWS Route 53 (DNS) + AWS Certificate Manager (SSL) ]
       │
       ▼
[ AWS Application Load Balancer (ALB) ]
       ├── /api/*, /docs, /health  ──────►  [ FastAPI Microservice (Port 8000) ]
       └── / (Default)             ──────►  [ Streamlit Dashboard (Port 8501) ]
                                                        │
                                                        ▼ (Direct Internal)
                                            [ Query Local Predictor Engine ]
```

---

## 3. Deployment Options & Cost Analysis

1. **Option A (Production Standard: ECS Fargate + ALB)**:
   - Serverless container orchestration with zero EC2 instances to manage.
   - Independent auto-scaling based on CPU utilization and request latency.
   - Isolated VPC private subnet execution for enterprise security.
   - Baseline cost: ~$45 - $85 / month.
2. **Option B (Serverless PaaS: AWS App Runner)**:
   - Direct connection to Amazon ECR with automatic deployment upon new image tag.
   - Integrated TLS/SSL certificate and load balancer.
   - Baseline cost: ~$25 - $40 / month.
3. **Option C (Single-Instance: AWS EC2 t3.medium + Docker Compose + NGINX)**:
   - Run both containers via Docker Compose on a single `t3.medium` instance.
   - NGINX handles SSL termination (Let's Encrypt Certbot) and WebSocket reverse proxying.
   - Ideal for low-cost staging and academic demonstration (~$15 - $25 / month).

---

## 4. Automation & Validation Evidence

1. **PowerShell and Bash Scripts Verified**:
   - Both `scripts/deploy_aws.sh` and `scripts/deploy_aws.ps1` parameterize AWS region, account ID, container tags, and cluster names.
2. **Unit Test Verification**:
   ```bash
   python -m unittest tests/test_production.py
   ```
   *Result*: `test_aws_deployment_artifacts_exist` confirmed that `docs/aws_deployment_guide.md`, `scripts/deploy_aws.sh`, `scripts/deploy_aws.ps1`, `infra/ecs-task-definition.json`, and `infra/nginx.conf` exist and match all expected specifications.

---

## 5. Next Steps

Proceed to **TASK 19 — Production Requirements & Health Endpoint Setup** for environment variable enforcement, structured logging, and observability.

# TASK 17 — Application Containerization (Docker)
## Completion Report

**Project**: Telecom Customer Churn Prediction & Retention System  
**Task ID**: TASK 17  
**Status**: COMPLETED  
**Executed On**: 2026-10-07  
**Assignment Requirement**: Requirement #15 ("Deployment - Use Docker and AWS") & Requirement #16 ("Dockerfile")

---

## 1. Objective

Package the FastAPI backend inference microservice and the interactive Streamlit Web UI into lightweight, secure, and reproducible Docker containers. Provide container orchestration via Docker Compose with health checks, environment isolation, and network bridging.

---

## 2. Containerization Architecture & Deliverables

### 2.1 Artifacts Created & Configured

| File | Purpose | Key Specifications |
|---|---|---|
| `Dockerfile.api` | Production container for FastAPI REST API | Python 3.10 slim base, `libgomp1` for LightGBM OpenMP, non-root user `appuser` (UID 1000), HTTP health check `/health`, port 8000 |
| `Dockerfile.ui` | Production container for Streamlit Web UI | Python 3.10 slim base, non-root `appuser`, Streamlit headless mode, HTTP health check `/_stcore/health`, port 8501 |
| `Dockerfile` | Multi-stage unified build file | Supports `--target api` and `--target ui` with shared cached base layer |
| `docker-compose.yml` | Multi-container orchestration | Bridges API and UI, dependency ordering with condition `service_healthy`, port mappings `8000:8000` & `8501:8501` |
| `.dockerignore` | Context optimization & security | Excludes virtualenvs (`venv`), `.git`, `__pycache__`, raw test logs, temporary checkpoints |

### 2.2 Container Topology

```
                       [ Host / External Client ]
                               │         │
                   Port 8000   │         │  Port 8501
                               ▼         ▼
┌──────────────────────────────────────────────────────────────┐
│                    Docker Bridge Network                     │
│                  (telecom-network)                           │
│                                                              │
│  ┌─────────────────────────┐     ┌─────────────────────────┐ │
│  │   telecom_churn_api     │     │   telecom_churn_ui      │ │
│  │   (FastAPI Microservice)│◄────┤   (Streamlit Dashboard) │ │
│  │   Port: 8000            │     │   Port: 8501            │ │
│  │   Healthcheck: /health  │     │   Health: /_stcore/     │ │
│  │   Non-root: appuser     │     │   Non-root: appuser     │ │
│  └─────────────────────────┘     └─────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Highlights & Hardening

1. **LightGBM Native Dependency Support**:
   Included `libgomp1` (GNU OpenMP runtime library) in `apt-get` packages on the Debian-slim base image to ensure high-performance multi-threaded tree inference without segmentation faults.
2. **Container Security (Non-Root User)**:
   Created and enforced an unprivileged `appuser` (UID/GID 1000) inside both container definitions, satisfying production compliance standards and preventing host filesystem breakout vulnerabilities.
3. **Automated Health Probing**:
   - API container executes `HEALTHCHECK` querying `curl -f http://localhost:8000/health` with a 15-second start period.
   - Streamlit container depends on `api` with `condition: service_healthy`, ensuring the UI never boots or queries a cold backend.
4. **Build Context Optimization**:
   Through `.dockerignore`, build context size was reduced from ~1.2 GB (containing local venvs and training cache) to < 10 MB, drastically improving build and pull speeds.

---

## 4. Validation Evidence

1. **Docker Compose Configuration Validation**:
   ```bash
   docker compose config
   ```
   *Result*: Schema validated cleanly with status code `0`, confirming valid service definitions, network bridges, environment maps, and health checks.
2. **Automated Verification Test**:
   ```bash
   python -m unittest tests/test_production.py
   ```
   *Result*: `test_docker_and_container_files_exist` and `test_docker_security_best_practices` **PASSED** cleanly.

---

## 5. Next Steps

Container images are fully configured and ready for cloud push and hosting under **TASK 18 — Cloud Deployment Architecture & AWS Hosting Plan**.

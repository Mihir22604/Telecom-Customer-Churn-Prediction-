import sys
from pathlib import Path

# Ensure repository root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

import time
from api.endpoints import router as api_router
from src.inference import get_predictor
from src.utils import get_logger, get_settings

logger = get_logger("api.main")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that pre-loads ML models into memory 
    before accepting incoming HTTP requests.
    """
    logger.info(f"Initializing Telecom Churn Prediction Service in [{settings.app_env}] environment...")
    try:
        predictor = get_predictor()
        logger.info(f"Model {predictor.metadata.get('model_name')} v{predictor.metadata.get('model_version')} successfully loaded.")
    except Exception as e:
        logger.error(f"Failed to load ML models during startup: {str(e)}")
        raise e
    yield
    logger.info("Shutting down Telecom Churn Prediction Service...")


# Initialize FastAPI application
app = FastAPI(
    title="Telecom Customer Churn Prediction & Retention API",
    description=(
        "Production RESTful API microservice exposing customer churn risk predictions, "
        "class probabilities, and actionable retention recommendations powered by a tuned LightGBM classifier."
    ),
    version=settings.model_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS Middleware
allowed_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Production request logging middleware measuring latency and tracking status codes."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    # Exclude high-frequency health probes from verbose logging if healthy
    client_ip = request.client.host if request.client else "unknown"
    response.headers["X-Process-Time"] = f"{duration_ms}ms"
    
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - "
        f"Latency: {duration_ms}ms - Client: {client_ip}"
    )
    return response

# Include API Router
app.include_router(api_router, prefix="/api/v1")
# Also mount at root for direct /predict and /health convenience
app.include_router(api_router)


@app.get("/", tags=["General"])
async def root():
    """Service landing endpoint with links to documentation and status."""
    return {
        "service": "Telecom Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "prediction_endpoint": "/predict",
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom error handler formatting Pydantic validation failures into clean JSON."""
    error_details = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error.get("loc", []))
        message = error.get("msg", "Invalid value")
        error_details.append({"field": field, "message": message})
        
    logger.warning(f"Validation error on {request.url.path}: {error_details}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Request Validation Error",
            "status_code": 422,
            "details": error_details
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Fallback handler catching unhandled server exceptions."""
    logger.error(f"Unhandled error processing {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "status_code": 500,
            "message": "An unexpected error occurred while processing the request."
        }
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=False)


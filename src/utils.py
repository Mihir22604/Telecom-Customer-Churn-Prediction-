import os
import sys
import json
import logging
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Load environment variables from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.
    Assumes this file is located at <project_root>/src/utils.py.
    """
    return Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Centralized strongly-typed application settings loaded from environment."""
    app_env: str = os.getenv("APP_ENV", "production")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    ui_port: int = int(os.getenv("UI_PORT", "8501"))
    fastapi_url: str = os.getenv("FASTAPI_URL", "http://localhost:8000")
    model_version: str = os.getenv("MODEL_VERSION", "1.0.0")
    model_path: str = os.getenv("MODEL_PATH", "models/best_churn_model.joblib")
    model_metadata_path: str = os.getenv("MODEL_METADATA_PATH", "models/model_metadata.json")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "standard")
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    secret_key: str = os.getenv("SECRET_KEY", "telecom-churn-retention-secret-key")


_SETTINGS_INSTANCE: Optional[Settings] = None


def get_settings() -> Settings:
    """Return singleton instance of application settings."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        _SETTINGS_INSTANCE = Settings()
    return _SETTINGS_INSTANCE


class JSONLogFormatter(logging.Formatter):
    """Structured JSON formatter for production log ingestion (Datadog, CloudWatch, etc.)."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def get_logger(name: str = "churn_system") -> logging.Logger:
    """
    Configure and return a standardized logger that outputs to console
    and logs/app.log, supporting standard or structured JSON formats.
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
        
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    if settings.log_format.lower() == "json":
        formatter = JSONLogFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    project_root = get_project_root()
    log_dir = project_root / "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_system_metrics() -> Dict[str, Any]:
    """
    Retrieve current machine metrics for health endpoint telemetry.
    """
    metrics = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1
    }
    
    try:
        import psutil
        vm = psutil.virtual_memory()
        process = psutil.Process()
        metrics.update({
            "memory_total_mb": round(vm.total / (1024 * 1024), 1),
            "memory_available_mb": round(vm.available / (1024 * 1024), 1),
            "memory_used_percent": vm.percent,
            "process_memory_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "process_cpu_percent": process.cpu_percent(interval=None)
        })
    except ImportError:
        metrics["memory_status"] = "psutil_not_installed"
    except Exception as e:
        metrics["metric_collection_error"] = str(e)
        
    return metrics

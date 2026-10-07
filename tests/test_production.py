import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.main import app
from src.utils import get_settings, get_system_metrics, get_project_root
from api.schemas import ChurnPredictionRequest


class TestProductionRequirements(unittest.TestCase):
    """
    Verification tests for Task 17, 18, and 19 production requirements:
    Environment configuration, health telemetry, container files, and schema validation.
    """
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.project_root = get_project_root()

    def test_settings_initialization(self):
        """Verify centralized settings load correctly with expected production defaults."""
        settings = get_settings()
        self.assertIsNotNone(settings.app_env)
        self.assertEqual(settings.api_port, 8000)
        self.assertEqual(settings.ui_port, 8501)
        self.assertIn("8000", settings.fastapi_url)
        self.assertEqual(settings.model_version, "1.0.0")

    def test_system_metrics_telemetry(self):
        """Verify system telemetry collects hardware and OS metrics."""
        metrics = get_system_metrics()
        self.assertIn("platform", metrics)
        self.assertIn("python_version", metrics)
        self.assertIn("cpu_count", metrics)
        self.assertGreaterEqual(metrics["cpu_count"], 1)

    def test_health_endpoint_production_payload(self):
        """Verify /health endpoint returns rich production telemetry."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_version"], "1.0.0")
        self.assertIn("uptime_seconds", data)
        self.assertIn("environment", data)
        self.assertIn("system_metrics", data)
        self.assertIsNotNone(data["system_metrics"])

    def test_docker_and_container_files_exist(self):
        """Verify all containerization and orchestration artifacts exist."""
        dockerfile = self.project_root / "Dockerfile"
        dockerfile_api = self.project_root / "Dockerfile.api"
        dockerfile_ui = self.project_root / "Dockerfile.ui"
        docker_compose = self.project_root / "docker-compose.yml"
        dockerignore = self.project_root / ".dockerignore"
        env_example = self.project_root / ".env.example"

        self.assertTrue(dockerfile.exists(), "Root Dockerfile missing")
        self.assertTrue(dockerfile_api.exists(), "Dockerfile.api missing")
        self.assertTrue(dockerfile_ui.exists(), "Dockerfile.ui missing")
        self.assertTrue(docker_compose.exists(), "docker-compose.yml missing")
        self.assertTrue(dockerignore.exists(), ".dockerignore missing")
        self.assertTrue(env_example.exists(), ".env.example missing")

    def test_docker_security_best_practices(self):
        """Verify Dockerfiles use non-root application users and healthchecks."""
        api_content = (self.project_root / "Dockerfile.api").read_text(encoding="utf-8")
        ui_content = (self.project_root / "Dockerfile.ui").read_text(encoding="utf-8")
        compose_content = (self.project_root / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("appuser", api_content)
        self.assertIn("HEALTHCHECK", api_content)
        self.assertIn("appuser", ui_content)
        self.assertIn("HEALTHCHECK", ui_content)
        self.assertIn("telecom-network", compose_content)
        self.assertIn("service_healthy", compose_content)

    def test_aws_deployment_artifacts_exist(self):
        """Verify AWS cloud deployment documentation and deployment scripts exist."""
        aws_guide = self.project_root / "docs" / "aws_deployment_guide.md"
        deploy_sh = self.project_root / "scripts" / "deploy_aws.sh"
        deploy_ps1 = self.project_root / "scripts" / "deploy_aws.ps1"
        ecs_task = self.project_root / "infra" / "ecs-task-definition.json"
        nginx_conf = self.project_root / "infra" / "nginx.conf"

        self.assertTrue(aws_guide.exists(), "AWS deployment guide missing")
        self.assertTrue(deploy_sh.exists(), "deploy_aws.sh missing")
        self.assertTrue(deploy_ps1.exists(), "deploy_aws.ps1 missing")
        self.assertTrue(ecs_task.exists(), "ecs-task-definition.json missing")
        self.assertTrue(nginx_conf.exists(), "nginx.conf missing")

    def test_strict_pydantic_bounds(self):
        """Verify boundary checking prevents invalid numbers or negative tenure."""
        valid_payload = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 10.0,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 80.0
        }

        # Valid payload works
        req = ChurnPredictionRequest(**valid_payload)
        self.assertEqual(req.tenure, 10.0)

        # Negative tenure raises error
        with self.assertRaises(Exception):
            invalid_payload = valid_payload.copy()
            invalid_payload["tenure"] = -5.0
            ChurnPredictionRequest(**invalid_payload)


if __name__ == "__main__":
    unittest.main()

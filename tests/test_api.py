import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import unittest
from fastapi.testclient import TestClient
from api.main import app


class TestFastAPIIntegration(unittest.TestCase):
    """
    End-to-end integration tests for FastAPI churn prediction REST service.
    """
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        
        # Standard test profiles
        cls.low_risk_payload = {
            "customerID": "TEST-LOW-01",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60.0,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 45.0,
            "TotalCharges": 2700.0
        }

        cls.high_risk_payload = {
            "customerID": "TEST-HIGH-01",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2.0,
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
            "MonthlyCharges": 95.0,
            "TotalCharges": 190.0
        }

    def test_root_endpoint(self):
        """Verify root landing endpoint returns 200 and valid links."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("documentation", data)

    def test_health_check(self):
        """Verify /health endpoint returns 200 OK and model is loaded."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertGreaterEqual(data["uptime_seconds"], 0.0)
        self.assertIn("model_version", data)

    def test_model_metadata(self):
        """Verify /metadata returns 28 feature names and test metrics."""
        resp = self.client.get("/metadata")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["feature_count"], 28)
        self.assertEqual(len(data["feature_names"]), 28)
        self.assertIn("test_set_metrics", data)
        self.assertGreater(data["test_set_metrics"]["Recall"], 0.70)

    def test_sample_profiles(self):
        """Verify /sample-profiles returns valid pre-configured test profiles."""
        resp = self.client.get("/sample-profiles")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("high_risk_customer", data)
        self.assertIn("low_risk_customer", data)

    def test_predict_single_customer_success(self):
        """Verify /predict returns 200 and schema-compliant response."""
        resp = self.client.post("/predict", json=self.high_risk_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["customerID"], "TEST-HIGH-01")
        self.assertIn(data["churn_prediction"], [0, 1])
        self.assertIn(data["churn_label"], ["Yes", "No"])
        self.assertTrue(0.0 <= data["churn_probability"] <= 1.0)
        self.assertIn(data["risk_level"], ["Low", "Medium", "High"])
        self.assertIsInstance(data["retention_recommendations"], list)
        self.assertGreater(len(data["retention_recommendations"]), 0)
        self.assertGreater(data["latency_ms"], 0.0)

    def test_risk_discrimination(self):
        """Verify that high-risk profile has higher churn probability than low-risk."""
        resp_low = self.client.post("/predict", json=self.low_risk_payload)
        resp_high = self.client.post("/predict", json=self.high_risk_payload)
        
        prob_low = resp_low.json()["churn_probability"]
        prob_high = resp_high.json()["churn_probability"]

        self.assertGreater(prob_high, prob_low, "High risk customer should have higher probability than low risk")
        self.assertEqual(resp_low.json()["risk_level"], "Low")
        self.assertEqual(resp_high.json()["risk_level"], "High")

    def test_predict_batch(self):
        """Verify /predict/batch endpoint executes multiple records properly."""
        batch_payload = {
            "customers": [self.low_risk_payload, self.high_risk_payload]
        }
        resp = self.client.post("/predict/batch", json=batch_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_processed"], 2)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertGreater(data["average_latency_ms"], 0.0)

    def test_predict_validation_error_negative_tenure(self):
        """Verify that negative tenure returns HTTP 422 with structured error details."""
        bad_payload = self.high_risk_payload.copy()
        bad_payload["tenure"] = -10.0
        resp = self.client.post("/predict", json=bad_payload)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertEqual(data["error"], "Request Validation Error")
        self.assertIn("details", data)

    def test_predict_validation_error_invalid_category(self):
        """Verify that invalid categorical value returns HTTP 422."""
        bad_payload = self.high_risk_payload.copy()
        bad_payload["Contract"] = "Infinite Contract"
        resp = self.client.post("/predict", json=bad_payload)
        self.assertEqual(resp.status_code, 422)

    def test_sequential_request_latency(self):
        """Verify that sequential predictions execute cleanly with low latency."""
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            resp = self.client.post("/predict", json=self.low_risk_payload)
            self.assertEqual(resp.status_code, 200)
            latencies.append((time.perf_counter() - t0) * 1000)
            
        avg_client_latency = sum(latencies) / len(latencies)
        self.assertLess(avg_client_latency, 250.0, f"Average latency too high: {avg_client_latency:.2f}ms")


if __name__ == '__main__':
    unittest.main()

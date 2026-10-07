import unittest
import pandas as pd
import numpy as np
from src.tuning import tune_logistic_regression

class TestTuning(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Smaller dataset for quick testing
        self.X_train = pd.DataFrame(np.random.rand(50, 5), columns=[f'F{i}' for i in range(5)])
        self.y_train = pd.Series(np.random.choice([0, 1], size=50))

    def test_tune_logistic_regression(self):
        # We only test LogisticRegression here to avoid long execution times of LightGBM tuning
        model, params, score = tune_logistic_regression(self.X_train, self.y_train)
        
        self.assertIsNotNone(model)
        self.assertIsInstance(params, dict)
        self.assertIn('C', params)
        self.assertIn('class_weight', params)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

if __name__ == '__main__':
    unittest.main()

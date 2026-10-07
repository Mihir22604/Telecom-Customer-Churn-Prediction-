import unittest
import pandas as pd
import numpy as np
from src.pipeline import split_data, get_cv_strategy

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create a mock dataframe
        np.random.seed(42)
        self.df = pd.DataFrame({
            'customerID': [f'CUST_{i}' for i in range(100)],
            'Feature_1': np.random.rand(100),
            'Feature_2': np.random.rand(100),
            'Churn': np.random.choice([0, 1], size=100, p=[0.75, 0.25])
        })

    def test_split_data(self):
        X_train, X_test, y_train, y_test = split_data(self.df, test_size=0.2, random_state=42)
        
        # Test shapes
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        
        # Ensure customerID is dropped
        self.assertNotIn('customerID', X_train.columns)
        self.assertNotIn('customerID', X_test.columns)
        
        # Test stratification (proportions should be similar)
        train_churn_ratio = y_train.mean()
        test_churn_ratio = y_test.mean()
        self.assertAlmostEqual(train_churn_ratio, test_churn_ratio, delta=0.1)

    def test_get_cv_strategy(self):
        cv = get_cv_strategy(n_splits=5)
        self.assertEqual(cv.n_splits, 5)

if __name__ == '__main__':
    unittest.main()

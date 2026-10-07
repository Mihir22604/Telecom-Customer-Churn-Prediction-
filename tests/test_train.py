import unittest
from src.train import get_models
from sklearn.base import is_classifier

class TestTrain(unittest.TestCase):
    def test_get_models(self):
        models = get_models()
        self.assertEqual(len(models), 7)
        for name, model in models.items():
            self.assertTrue(is_classifier(model), f"{name} is not a valid classifier")
            
        self.assertIn('Logistic Regression', models)
        self.assertIn('Decision Tree', models)
        self.assertIn('Random Forest', models)
        self.assertIn('XGBoost', models)
        self.assertIn('LightGBM', models)
        self.assertIn('SVM', models)
        self.assertIn('KNN', models)

if __name__ == '__main__':
    unittest.main()

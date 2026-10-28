
import unittest
from unittest.mock import patch

import app as prediction_app


class TestPredictionAPI(unittest.TestCase):

    def setUp(self):
        prediction_app.app.config["TESTING"] = True
        self.client = prediction_app.app.test_client()

    def test_home_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_missing_json_body(self):
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_missing_required_fields(self):
        response = self.client.post(
            "/predict",
            json={"attendance": 90}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    @patch("app.load_model")
    def test_successful_prediction(self, mock_load_model):
        class FakeModel:
            def predict(self, data):
                self.assert_columns = list(data.columns)
                return [1]

        mock_load_model.return_value = FakeModel()

        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 80,
                "assignment_marks": 85,
                "previous_score": 75
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "PASS")
        self.assertEqual(response.get_json()["prediction_code"], 1)

    @patch("app.load_model")
    def test_fail_prediction(self, mock_load_model):
        class FakeModel:
            def predict(self, data):
                return [0]

        mock_load_model.return_value = FakeModel()

        response = self.client.post(
            "/predict",
            json={
                "attendance": 55,
                "internal_marks": 40,
                "assignment_marks": 45,
                "previous_score": 35
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "FAIL")
        self.assertEqual(response.get_json()["prediction_code"], 0)
    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")
    return accuracy

if __name__ == "__main__":
    unittest.main()

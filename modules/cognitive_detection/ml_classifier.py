import joblib
import numpy as np
import os

MODEL_PATH   = "models/cognitive_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

class MLClassifier:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run train_model.py first.")

        self.model   = joblib.load(MODEL_PATH)
        self.encoder = joblib.load(ENCODER_PATH)
        print("ML model loaded successfully!")
        print(f"Classes: {self.encoder.classes_}")

    def predict(self, ear, pitch_adj, yaw_adj,
                blink_rate, closed_frames):
        """
        Takes extracted features and returns predicted state label.
        """
        features = np.array([[
            ear, pitch_adj, yaw_adj,
            blink_rate, closed_frames
        ]])
        prediction = self.model.predict(features)[0]
        label      = self.encoder.inverse_transform([prediction])[0]
        confidence = self.model.predict_proba(features)[0].max()
        return label, round(confidence, 3)
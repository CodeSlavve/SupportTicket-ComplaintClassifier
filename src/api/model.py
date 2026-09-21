import pickle
from pathlib import Path


class ModelService:

    def __init__(self, pipeline_path: str, encoder_path: str):
        self.pipeline_path = Path(pipeline_path)
        self.encoder_path = Path(encoder_path)

        self.pipeline = None
        self.label_encoder = None

    def load(self):
        with open(self.pipeline_path, "rb") as f:
            self.pipeline = pickle.load(f)

        with open(self.encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

    def predict(self, text: str):

        prediction = self.pipeline.predict([text])[0]

        probabilities = self.pipeline.predict_proba([text])[0]

        confidence = float(probabilities.max())

        category = self.label_encoder.inverse_transform(
            [prediction]
        )[0]

        return category, confidence

    def predict_batch(self, texts: list[str]):
        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)

        categories = self.label_encoder.inverse_transform(predictions)

        results = []

        for category, probability in zip(categories, probabilities):
            confidence = float(probability.max())

            results.append({
                "category": category,
                "confidence": confidence
            })

        return results
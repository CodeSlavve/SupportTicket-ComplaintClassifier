FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/api ./src/api
COPY models/xgb_pipeline.pkl ./models/xgb_pipeline.pkl
COPY models/label_encoder.pkl ./models/label_encoder.pkl

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
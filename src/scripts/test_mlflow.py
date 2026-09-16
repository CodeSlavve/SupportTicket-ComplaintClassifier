import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run():
    mlflow.log_param("model", "dummy-model")
    mlflow.log_param("dataset", "dummy-dataset")
    mlflow.log_metric("accuracy", 0.95)

print("Dummy MLflow run completed!")
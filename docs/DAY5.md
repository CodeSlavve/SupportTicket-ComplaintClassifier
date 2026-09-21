# Day 5 — FastAPI Inference API

## Objective

The objective of Day 5 was to convert the finalized XGBoost complaint classifier into a clean, self-contained local REST API using **FastAPI**.

The API was designed to load the trained model artifacts at startup, validate incoming complaint text using **Pydantic**, expose prediction endpoints, and provide automated API tests using **pytest** and `TestClient`.

The service was also structured with deployment in mind so that it does not depend on notebook-specific paths or the current working directory.

---

# 1. FastAPI Service

The API was organized into separate modules:

```text
src/
└── api/
    ├── main.py
    ├── model.py
    └── schemas.py
```

Each module has a specific responsibility:

* `schemas.py` — request and response validation using Pydantic;
* `model.py` — model loading and inference;
* `main.py` — FastAPI application and API endpoints.

This keeps the inference service separate from the notebooks and model development code.

---

# 2. Model Loading

The API uses the finalized production artifacts:

```text
models/
├── xgb_pipeline.pkl
└── label_encoder.pkl
```

The XGBoost pipeline contains the TF-IDF vectorizer and classifier, allowing the API to accept raw complaint text directly.

The model is loaded once when the FastAPI application starts using the application's lifespan.

This prevents the model from being loaded repeatedly for individual requests and keeps the loaded artifacts available throughout the server process.

---

# 3. Request Validation

Pydantic was used to define the API input and output contracts.

For single prediction requests, the API validates:

* complaint text is provided;
* input is a string;
* text is not empty;
* text does not exceed the configured character limit.

Batch prediction additionally validates:

* the batch contains at least one complaint;
* the batch does not exceed 600 complaints;
* every complaint is a string;
* every complaint satisfies the configured character limit.

Invalid requests are rejected by FastAPI/Pydantic before reaching the ML model.

---

# 4. Prediction Endpoints

The service provides three endpoints:

| Method | Endpoint         | Purpose                                    |
| ------ | ---------------- | ------------------------------------------ |
| GET    | `/health`        | API health check                           |
| POST   | `/predict`       | Predict the category of one complaint      |
| POST   | `/predict/batch` | Predict categories for multiple complaints |

The single prediction response contains:

```text
category
confidence
```

The batch endpoint returns the corresponding prediction and confidence for each complaint.

---

# 5. Batch Inference

Batch prediction was added to allow multiple complaints to be passed to the model together rather than making a separate model call for every complaint.

The inference flow is:

```text
Multiple Complaints
        ↓
Pydantic Validation
        ↓
TF-IDF + XGBoost Pipeline
        ↓
Predictions
        ↓
Label Encoder
        ↓
Categories + Confidence
```

The batch endpoint supports up to **600 complaints per request**.

---

# 6. API Testing

Automated API tests were implemented using:

```text
pytest
FastAPI TestClient
```

The tests cover the main service behavior, including:

* `/health` endpoint;
* valid `/predict` requests;
* invalid prediction input;
* batch prediction;
* batch input validation.

The completed API test suite passes successfully.

---

# 7. Structured API Logging

Structured application logging was added to capture useful inference metadata without storing the actual complaint text.

The logging layer records:

* prediction endpoint;
* input length;
* predicted category;
* confidence;
* inference latency;
* batch size for batch predictions;
* model loading and startup failures.

Logs are written to:

```text
logs/
└── api.log
```

The complaint text itself is **not logged**, avoiding unnecessary storage of potentially sensitive user information.

A centralized logger was added in:

```text
src/api/logger.py
```

The logger writes to both the API console and the `logs/api.log` file.

Example prediction log:

```text
prediction | endpoint=/predict | input_length=487 |
category=Credit card | confidence=0.9132 | latency_ms=15.42
```

Batch requests record the batch size and total inference latency.

The logging layer also records model loading failures during application startup, allowing deployment failures to be detected immediately rather than producing repeated prediction-time errors.

---

# 8. Deployment-Ready Path Handling

Model paths are resolved from the project root using `pathlib` rather than relying on the directory from which the application is launched.

The API therefore does not depend on paths such as:

```text
../models/
../../models/
```

or on running the application from inside the `notebooks/` directory.

This makes the service self-contained and prepares it for the Docker packaging stage.

---

# 9. Day 5 Outcome

By the end of Day 5, the project had:

* converted the production XGBoost model into a FastAPI inference service;
* implemented `/health`;
* implemented `/predict`;
* implemented `/predict/batch`;
* added Pydantic request/response validation;
* added batch input validation;
* implemented startup model loading;
* added automated API tests using `pytest` and `TestClient`;
* removed notebook-specific path assumptions;
* verified that the API test suite passes.

The project has now moved from an offline trained model into a reusable local inference service.

---

# 10. Transition to Day 6

With the core API working and tested, the next stage can focus on **hardening the service** and improving its reliability before moving toward data/model versioning and containerization.

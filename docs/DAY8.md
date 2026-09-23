# Day 8 — Deployment

## Objective

Deploy the containerized FastAPI complaint classification service as a live, publicly reachable API and confirm that production requests use the deployed model successfully.

This marks the transition from a locally tested and CI-built service to a working deployment that can be accessed through a public URL.

---

## 1. Deployment Platform

The service was deployed on **Render** using the production Docker image created for the project.

Render hosts the containerized FastAPI application and exposes it as a public web service. The deployment makes the same API that was tested locally and inside Docker available outside the development environment.

---

## 2. Live Application URL

The deployed API is publicly available at:
https://supportticket-complaintclassifier.onrender.com


The interactive FastAPI documentation is available at:
https://supportticket-complaintclassifier.onrender.com/docs


The `/docs` page provides a browser-based interface for inspecting and testing the API endpoints, including `/health`, `/predict`, and `/predict/batch`.

---

## 3. Production API Verification

The deployed service successfully handled prediction requests in production.

An example production prediction log was recorded as follows:

```text
2026-09-23 16:46:38,072 | INFO | prediction | endpoint=/predict | input_length=83 | category=Mortgage | confidence=0.6250 | latency_ms=84.14
```

This confirms that the live `/predict` endpoint:

- Received the complaint text.
- Loaded and used the deployed classification model.
- Returned the predicted category: `Mortgage`.
- Returned a confidence score of `0.6250`.
- Logged the request latency and useful prediction metadata.

---

## 4. Prediction Logging

Prediction logging is enabled for the deployed API.

Each prediction request records:

- Timestamp
- Log level
- Endpoint
- Input text length
- Predicted complaint category
- Confidence score
- Request latency in milliseconds

The production log entry provides lightweight operational visibility without logging the full complaint text, which keeps the logs useful while avoiding unnecessary exposure of user input.

---

## 5. Latency Comparison

Latency was measured for both the local service and the Render deployment.

| Environment | Prediction latency |
| --- | ---: |
| Local API | 56.25 ms |
| Render production API | 84.14 ms |

The production request was approximately **27.89 ms** slower than the local request.

This difference is expected because a live deployment includes network travel and cloud-hosting overhead in addition to model inference. The measured production request completed in 84.14 ms.

---

## 6. Render Health Check

Render is configured to use the API health endpoint as its health-check path:

```text
/health
```

This endpoint provides Render with a simple application-status check for the deployed FastAPI service.

---

## 7. Deployment Outcome

The complaint classification API is now deployed as a live Render service and can be accessed through its public documentation page:

```text
https://supportticket-complaintclassifier.onrender.com/docs
```

The deployed service has been verified through a real production prediction request, and prediction logging captures the category, confidence, input length, timestamp, and latency for each request.

The complete deployment flow is:

```text
Dockerized FastAPI service
        ↓
Render deployment
        ↓
Public FastAPI endpoint
        ↓
Live prediction request
        ↓
Prediction log with latency and confidence
```

---
**Live API documentation:**
```text
https://supportticket-complaintclassifier.onrender.com/docs
```

**Measured production prediction latency:** `84.14 ms`

**Measured local prediction latency:** `56.25 ms`

The project now demonstrates the complete path from model development and evaluation to API implementation, containerization, CI/CD, and cloud deployment.
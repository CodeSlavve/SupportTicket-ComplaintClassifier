# Day 7 — Containerization & CI/CD

## Objective

Containerize the FastAPI complaint classification service and automate testing, linting, Docker image building, and image publishing using GitHub Actions.

The final workflow ensures that the application is tested before a Docker image is built and that successful builds are published to GitHub Container Registry (GHCR) with traceable image tags.

---

## 1. Docker Containerization

The FastAPI service is packaged into a lightweight Docker image containing only the files and dependencies required to run the production API.

### Runtime Components

The container requires:

- `src/api/`
- `models/xgb_pipeline.pkl`
- `models/label_encoder.pkl`
- `requirements-api.txt`

Development-only resources such as notebooks, datasets, results, MLflow artifacts, and the DistilBERT directory are excluded from the image.

### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/api ./src/api
COPY models/xgb_pipeline.pkl ./models/xgb_pipeline.pkl
COPY models/label_encoder.pkl ./models/label_encoder.pkl

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

The dependency file is copied and installed before the application source code so Docker can reuse the dependency layer when only application code changes.

### `.dockerignore`

The Docker build context excludes files that are not required by the production API:

```text
venv
.venv
data/
notebooks/
docs/
mlruns/
mlflow.db
distilbert/
mlartifacts/
```

This keeps the image smaller and prevents development and training artifacts from being included in the production container.

---

## 2. Local Docker Build

The image was built locally using:

```bash
docker build -t ticket-classifier .
```

The build completed successfully using the production Dockerfile and required model artifacts.

---

## 3. Running the Container

The container was started with:

```bash
docker run -p 8000:8000 ticket-classifier
```

The port mapping exposes the FastAPI service running on port `8000` inside the container to port `8000` on the host machine.

The API runs with:

```text
0.0.0.0:8000
```

so that it is accessible outside the container.

---

## 4. Container API Verification

The running container was tested using the same API endpoints used during local FastAPI testing.

### Health Check

```text
GET /health
```

Expected response:

```json
{
  "status": "ok"
}
```

### Single Prediction

```text
POST /predict
```

The endpoint successfully accepted complaint text and returned:

- Predicted complaint category
- Confidence score

### Batch Prediction

```text
POST /predict/batch
```

The endpoint successfully processed multiple complaint texts and returned predictions for each input.

The containerized API produced the expected predictions using the same XGBoost pipeline and label encoder used by the local FastAPI service.

---

## 5. Docker Image Size

The image size was checked using:

```bash
docker images
```

The final image contains the FastAPI runtime, XGBoost/scikit-learn dependencies, and the required model artifacts without including the training dataset or other development resources.

**Image size:** 499MB

---

## 6. GitHub Actions CI/CD

After verifying the Docker container locally, a GitHub Actions workflow was created at:

```text
.github/workflows/ci.yml
```

The workflow automates three stages:

```text
Test
  ↓
Lint
  ↓
Build + Push
```

The Docker build depends on both the Test and Lint jobs passing successfully.

---

## 7. Test Job

The Test job:

1. Checks out the repository.
2. Sets up Python 3.13.
3. Installs project dependencies from `requirements.txt`.
4. Configures the DVC Google Drive remote.
5. Pulls the DVC-tracked artifacts.
6. Runs the complete pytest suite.

The DVC artifacts are pulled during CI because the model `.pkl` files are tracked through DVC rather than stored directly in Git.

The Google Drive credentials are supplied through GitHub repository secrets.

The tests verify the FastAPI endpoints, including:

- `/health`
- `/predict`
- `/predict/batch`
- Empty prediction input validation
- Empty batch validation
- Maximum text-length validation

---

## 8. Lint Job

Ruff is used to check the source code.

The workflow runs:

```bash
ruff check src/
```

This checks the Python source code for linting and code-quality issues before the Docker image is built.

---

## 9. Docker Build and GHCR Publishing

The Build job runs only after both Test and Lint succeed.

It:

1. Checks out the repository.
2. Sets up Python 3.13.
3. Installs the required DVC Google Drive dependencies.
4. Pulls the DVC model artifacts.
5. Authenticates with GitHub Container Registry.
6. Builds the Docker image using the existing Dockerfile.
7. Pushes the image to GHCR.

The image is published as:

```text
ghcr.io/codeslavve/supportticket-complaintclassifier
```

---

## 10. Image Tagging

Each successful main-branch build produces two image tags:

```text
latest
```

and:

```text
<git commit SHA>
```

The workflow uses:

```yaml
tags: |
  ${{ env.IMAGE_NAME }}:latest
  ${{ env.IMAGE_NAME }}:${{ github.sha }}
```

The `latest` tag provides a convenient reference to the most recently published image, while the commit SHA identifies the exact Git revision that produced the image.

This makes it possible to trace a deployed image back to its source code revision.

---

## 11. Pull Request Behaviour

The workflow runs for pull requests targeting `main`.

For pull requests:

```text
Test  → runs
Lint  → runs
Build → skipped
```

The Docker image is only built and pushed when code is pushed to `main`.

This prevents pull requests from publishing Docker images to GHCR before they are merged.

---

## 12. CI Failure Verification

The pipeline was deliberately tested with a failing test to verify that CI does not simply report success.

With the test intentionally broken:

```text
Test  → FAILED
Lint  → PASSED
Build → SKIPPED
```

Because the Build job has:

```yaml
needs:
  - test
  - lint
```

a failed test prevents the Docker build and publishing stage from running.

After restoring the test, the workflow returned to a successful state.

---

## 13. Successful CI Run

The final GitHub Actions workflow successfully completed:

```text
Test                         PASSED
Lint                         PASSED
Build + Push Docker Image    PASSED
```

The resulting Docker image was published to GHCR with both the `latest` and Git commit SHA tags.

**GHCR package:** `ghcr.io/codeslavve/supportticket-complaintclassifier`

---

## 14. Final Outcome

The FastAPI complaint classification service is now containerized and connected to an automated CI/CD pipeline.

The complete flow is:

```text
Developer pushes code
        ↓
GitHub Actions
        ↓
Install dependencies
        ↓
Pull DVC artifacts
        ↓
Run pytest
        ↓
Run Ruff
        ↓
Build Docker image
        ↓
Push image to GHCR
        ↓
Tagged with latest + Git SHA
```

The resulting GHCR image was published successfully and provides the container image used for the deployment stage.
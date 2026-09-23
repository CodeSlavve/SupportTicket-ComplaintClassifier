# CFPB Complaint Classifier

An end-to-end NLP classification system that classifies consumer complaint narratives into **8 CFPB product categories** using a balanced dataset of **50,000 complaints**. The project compares **Logistic Regression, XGBoost, and DistilBERT**, selects **XGBoost** for production based on its performance/resource tradeoff, and serves the final model through a **FastAPI API** packaged with **Docker**, versioned with **DVC**, validated and published through **GitHub Actions + GHCR**, and deployed as a live service on **Render**.

---

## Live Demo

**Live API:**  
https://supportticket-complaintclassifier.onrender.com

**Swagger UI:**  
https://supportticket-complaintclassifier.onrender.com/docs

The Swagger interface provides an interactive way to inspect and test the deployed API.

Available endpoints:

```text
GET  /health
POST /predict
POST /predict/batch
```

---

## Architecture

### ML Pipeline

```text
Complaint Text
      ↓
Text Preprocessing
      ↓
TF-IDF
      ↓
XGBoost
      ↓
Complaint Category
      +
Confidence Score
```

The production artifact `models/xgb_pipeline.pkl` contains the TF-IDF transformation and XGBoost classifier. The label encoder is stored separately in `models/label_encoder.pkl`.

### Deployment Pipeline

```text
Code
 ↓
GitHub
 ↓
GitHub Actions
 ├── Tests
 ├── Ruff
 └── Docker Build
       ↓
      GHCR
       ↓
     Render
       ↓
  Live FastAPI API
```

---

## Results

### Final XGBoost Test Performance

| Metric | Score |
|---|---:|
| Accuracy | **83.23%** |
| Macro Precision | **83.37%** |
| Macro Recall | **83.23%** |
| Macro F1 | **83.26%** |
| Weighted F1 | **83.26%** |

### Model Comparison

| Model | Validation Macro F1 | Model Size | RAM Usage | Inference |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.94% | 1.99 MB | 22.71 MB | 0.19 ms |
| **XGBoost** | **83.35%** | **6.33 MB** | **88.77 MB** | **0.55 ms** |
| DistilBERT | 85.09% | 256.12 MB | 703.36 MB | 0.82 ms |

DistilBERT achieved a **1.74 percentage-point higher validation Macro F1** than XGBoost, but required substantially more memory.

XGBoost was selected for production based on the combined performance and resource tradeoff.

---

## Error Analysis / Limitations

### Error Rate by Complaint Length

| Complaint Length | Error Rate |
|---|---:|
| 1–25 characters | 31.12% |
| 26–50 characters | 25.33% |
| 51–100 characters | 19.06% |
| 101–200 characters | 15.33% |
| 201–500 characters | 14.69% |
| 500+ characters | 10.71% |

Short complaints produced higher error rates, while longer narratives generally provided more information for classification.

The largest validation confusion was:

```text
Checking or savings account → Money transfer: 160
Money transfer → Checking or savings account: 153
```

### Limitations

- The model predicts only the eight categories used during training.
- Short or vague complaints are more difficult to classify.
- Some categories contain overlapping terminology.
- Results are based on a 50,000-complaint sample and may not generalize to other datasets or future complaint distributions.
- Confidence scores should not be interpreted as guaranteed probabilities of correctness.
- The model is intended for complaint categorization and does not replace human review or make financial or legal decisions.

For detailed model information, see [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Quick Start

### Clone

```bash
git clone https://github.com/CodeSlavve/SupportTicket-ComplaintClassifier.git
cd SupportTicket-ComplaintClassifier
```

### Create Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

For the complete project:

```bash
pip install -r requirements.txt
```

For API-only usage: 

> *requirements-api.txt contains only the minimal dependencies required to run the deployed FastAPI service, while requirements.txt includes the full training and experimentation stack (including libraries like PyTorch and transformers) used for model development and notebooks, keeping the production API lean and separate from development workloads.*

```bash
pip install -r requirements-api.txt
```

### Run the API

```bash
uvicorn src.api.main:app --reload
```

Local API:

```text
http://localhost:8000
```

Local Swagger UI:

```text
http://localhost:8000/docs
```

---

## Reproduction

### Reproducing the Dataset and Model Pipeline

To reproduce the project from a clean checkout, first generate the dataset with:

```bash
python src/scripts/sample_cfpb.py
```

This script creates the local CFPB sample dataset used throughout the project. After the dataset is available, run the notebook sequence in order to reproduce the model workflow:

```
notebooks/day1-EDA.ipynb
notebooks/day2-baseline-modeling-pipline.ipynb
notebooks/day3-distilbert.ipynb
notebooks/day4-error-analysis.ipynb
```

Following this order recreates the exploratory analysis, classical baselines, transformer experiment, and error analysis used to produce the final production artifacts.

> Suggestion: the DistilBERT training notebook is best run in Google Colab because the fine-tuning workflow is GPU-intensive and benefits from the additional memory and runtime available there.

> This reproduction path does not require access to the project's DVC remote. The dataset is generated locally by sample_cfpb.py, while DVC remains used for versioning and CI/CD artifact retrieval.

### DVC

The production dataset and model artifacts are versioned with DVC.

Tracked artifacts include:

```text
data/cfpb_sample_50k.csv
models/xgb_pipeline.pkl
models/label_encoder.pkl
```

Install DVC with Google Drive support:

```bash
pip install dvc dvc-gdrive
```

Restore the tracked artifacts:

```bash
dvc pull
```

DVC credentials are kept outside version control. You can use the DVC method to reproduce the Data/model versioning.

### Docker

Build the production container:

```bash
docker build -t ticket-classifier .
```

Or directly pull the created docker image:

```bash
docker pull ghcr.io/codeslavve/supportticket-complaintclassifier:latest
```

Run it:

```bash
docker run -p 8000:8000 ticket-classifier
```

Then open:

```text
http://localhost:8000/docs
```

### Tests

```bash
pytest
```

### Linting

```bash
ruff check src/
```

### CI/CD

GitHub Actions automatically runs tests and linting on repository changes.

For successful pushes to `main`, the workflow also:

1. Restores required DVC artifacts.
2. Builds the Docker image.
3. Publishes the image to GHCR.
4. Tags the image with `latest` and the Git commit SHA.

Container image:

```text
ghcr.io/codeslavve/supportticket-complaintclassifier
```

---

## API Example

### `POST /predict`

Example request:

```json
{
  "text": "I have an issue with my mortgage payment and need help resolving it."
}
```

Example response:

```json
{
  "category": "Mortgage",
  "confidence": 0.625
}
```

### `POST /predict/batch`

The batch endpoint accepts multiple complaint texts and returns predictions for each input.

The API validates request data and limits the maximum batch size to **600 items**.

### `GET /health`

Used to verify that the API service is running.

---

## Tech Stack

### Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- PyTorch
- Transformers
- Hugging Face Datasets

### NLP

- Text preprocessing
- TF-IDF
- Unigrams + bigrams
- DistilBERT benchmarking

### Experiment Tracking

- MLflow

### API

- FastAPI
- Pydantic
- Uvicorn

### Testing & Code Quality

- pytest
- Ruff

### Data & Model Versioning

- DVC
- Google Drive

### Deployment & Infrastructure

- Docker
- GitHub Actions
- GitHub Container Registry
- Render

---

## Project Structure

```text
SupportTicket-ComplaintClassifier/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dvc/
│
├── data/
│   └── cfpb_sample_50k.csv
│
├── docs/
│   ├── DAY1.md
│   ├── DAY2.md
│   ├── DAY3.md
│   ├── DAY4.md
│   ├── DAY5.md
│   ├── DAY6.md
│   ├── DAY7.md
│   ├── DAY8.md
│   ├── project_writeup.md
│   └── MODEL_CARD.md
│
├── models/
│   ├── xgb_pipeline.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   ├── day0-sanity-check.ipynb
│   ├── day1-EDA.ipynb
│   ├── day2-baseline-modeling-pipline.ipynb
│   ├── day3-distilbert.ipynb
│   └── day4-error-analysis.ipynb
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── schemas.py
│   │   └── logger.py
│   ├── scripts/
│   │   └── sample_cfpb.py
│   └── __init__.py
│
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements-api.txt
└── README.md
```

*Some dataset, model, experiment, and generated files are managed through DVC or excluded from Git.*

---

## Documentation

Detailed documentation is available in [`docs/`](docs/).

| Document | Description |
|---|---|
| [`DAY1.md`](docs/DAY1.md) | Dataset preparation and preprocessing |
| [`DAY2.md`](docs/DAY2.md) | Classical ML model development |
| [`DAY3.md`](docs/DAY3.md) | DistilBERT experiment |
| [`DAY4.md`](docs/DAY4.md) | Model comparison and error analysis |
| [`DAY5.md`](docs/DAY5.md) | FastAPI implementation |
| [`DAY6.md`](docs/DAY6.md) | DVC data and model versioning |
| [`DAY7.md`](docs/DAY7.md) | Docker and CI/CD |
| [`DAY8.md`](docs/DAY8.md) | Render deployment |
| [`MODEL_CARD.md`](docs/MODEL_CARD.md) | Model details, capabilities, performance, and limitations |
| [`project_writeup.md`](project_writeup.md) | Complete project development write-up |

---

## Live Project

**API:**  
https://supportticket-complaintclassifier.onrender.com

**Swagger:**  
https://supportticket-complaintclassifier.onrender.com/docs

**Source Code:**  
https://github.com/CodeSlavve/SupportTicket-ComplaintClassifier
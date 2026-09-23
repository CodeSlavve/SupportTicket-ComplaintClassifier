# CFPB Consumer Complaint Classifier

## 1. Project Overview

The **CFPB Consumer Complaint Classifier** is an 8-class text classification system designed to automatically categorize consumer financial complaints into their corresponding product categories.

The project uses complaint narratives from the **Consumer Financial Protection Bureau (CFPB)** and applies natural language processing and machine learning techniques to classify complaints.

Two modeling approaches were explored:

1. **Classical NLP:** TF-IDF + machine learning
2. **Transformer NLP:** Fine-tuned DistilBERT

The classical approach evaluated Logistic Regression and XGBoost, while DistilBERT was fine-tuned as the transformer-based approach.

The project followed an end-to-end workflow covering data preparation, model development, validation, model comparison, error analysis, experiment tracking, final model selection, API development, containerization, CI/CD, and deployment.

The final production model is an **XGBoost classifier using TF-IDF features**, exposed through a FastAPI service and deployed using Docker on Render.

---

# 2. Problem Statement

Consumer complaints contain valuable information about the financial products and services involved in a customer's issue. However, manually categorizing large volumes of complaints can be time-consuming.

The objective of this project is to build a machine learning system that takes a consumer complaint narrative as input and predicts the corresponding CFPB product category.

### Input

A cleaned consumer complaint narrative.

### Output

One of eight predefined CFPB product categories.

This is formulated as an **8-class multi-class text classification problem**.

---

# 3. Objectives

The main objectives of the project were:

- Prepare a clean dataset from CFPB consumer complaint data.
- Select a focused set of financial product categories.
- Create a balanced dataset for classification.
- Establish a classical NLP baseline using TF-IDF.
- Compare Logistic Regression and XGBoost.
- Fine-tune a pretrained DistilBERT model.
- Compare classical NLP with transformer-based NLP.
- Evaluate candidate models using a consistent validation set.
- Evaluate the selected model on an untouched test set.
- Analyze model errors and identify recurring failure patterns.
- Track experiments and artifacts using MLflow.
- Version the dataset and model artifacts using DVC.
- Package the selected model into an inference pipeline.
- Build a FastAPI prediction service.
- Containerize the API using Docker.
- Automate testing, linting, image building, and publishing using GitHub Actions.
- Deploy the service to Render.

---

# 4. Dataset

The source data was obtained from the **Consumer Financial Protection Bureau consumer complaint database**.

The complete CFPB dataset is substantially larger than what was required for this project. A representative working sample was therefore created from the available complaint data.

After filtering the relevant complaints and selecting the eight target product categories, a balanced dataset containing **50,000 complaints** was created.

The sampled dataset contains the original complaint fields required during the preparation stage. During preprocessing, the data was reduced to the two fields required for model training:

```text
clean_complaint_text
Product
```

Where:

- `clean_complaint_text` is the processed complaint narrative used as model input.
- `Product` is the target class.

The distinction between the sampled dataset and the processed modeling data is important: the **50,000-row sample is the working source dataset**, while the preprocessing stage produces the two-column representation used for model training.

---

# 5. Selected Product Categories

Eight product categories were finalized for the classification task:

1. Checking or savings account
2. Credit card
3. Credit reporting or other personal consumer reports
4. Debt collection
5. Money transfer, virtual currency, or money service
6. Mortgage
7. Payday loan, title loan, personal loan, or advance loan
8. Vehicle loan or lease

The dataset was balanced so that each category contained exactly **6,250 complaints**.

| Product Category | Samples |
|---|---:|
| Checking or savings account | 6,250 |
| Credit card | 6,250 |
| Credit reporting or other personal consumer reports | 6,250 |
| Debt collection | 6,250 |
| Money transfer, virtual currency, or money service | 6,250 |
| Mortgage | 6,250 |
| Payday loan, title loan, personal loan, or advance loan | 6,250 |
| Vehicle loan or lease | 6,250 |
| **Total** | **50,000** |

Each class therefore represents **12.5%** of the final dataset.

---

# 6. Data Preprocessing

The original CFPB complaint data contains many fields that are not required for this text classification task.

The data preparation process filtered the required product categories, removed complaints without usable complaint narratives, and produced cleaned complaint text for downstream modeling.

The resulting modeling data contains:

```text
clean_complaint_text
Product
```

The `clean_complaint_text` field is used as the input to the NLP models, while `Product` is the target label.

The preprocessing and dataset preparation stages also produced the train, validation, and test datasets used consistently throughout the modeling workflow.

---

# 7. Dataset Split

The 50,000 samples were divided into training, validation, and test sets.

| Split | Samples | Percentage |
|---|---:|---:|
| Training | 35,000 | 70% |
| Validation | 7,500 | 15% |
| Test | 7,500 | 15% |
| **Total** | **50,000** | **100%** |

The same overall split was used for the classical NLP experiments and the transformer experiment.

The validation set was used during model development and comparison.

The test set was kept untouched until the final XGBoost model had been selected. It was then used once for the final evaluation.

---

# 8. Approach 1 — Classical NLP

## 8.1 TF-IDF

The first modeling approach used **Term Frequency–Inverse Document Frequency (TF-IDF)** to convert complaint narratives into numerical feature vectors.

The vectorizer was fitted on the training data and then used to transform the validation and test sets.

The main configuration included:

```text
max_features = 20,000
ngram_range = (1, 2)
min_df = 5
```

Using `(1, 2)` allowed the representation to include both individual words and two-word combinations.

The resulting training feature matrix used up to 20,000 TF-IDF features.

The same TF-IDF representation was used for both Logistic Regression and XGBoost so that the classical models could be compared under the same feature representation.

---

# 9. Logistic Regression

Logistic Regression was used as the first classical machine learning baseline.

It was trained using the TF-IDF representation and evaluated on the validation set.

The model provided a lightweight baseline for the multi-class classification problem.

Its validation Macro F1 score was:

```text
82.94%
```

The trained Logistic Regression pipeline had an approximate serialized size of **1.99 MB**, used approximately **22.71 MB RAM** during the measured inference test, and had an average inference time of approximately **0.19 ms per complaint** in the local benchmark.

---

# 10. XGBoost

XGBoost was evaluated as the second classical machine learning model.

It operated on the same TF-IDF representation used by Logistic Regression.

The main configuration was:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.1
random_state = 42
n_jobs = -1
```

Using the same dataset split and TF-IDF representation allowed XGBoost to be compared directly with the Logistic Regression baseline.

XGBoost achieved a validation Macro F1 score of:

```text
83.35%
```

The measured deployment characteristics were approximately:

```text
Model size:       6.33 MB
RAM usage:        88.77 MB
Inference time:   0.55 ms per complaint
```

XGBoost was subsequently carried forward as the classical model for comparison with DistilBERT.

---

# 11. Approach 2 — Fine-Tuned DistilBERT

To evaluate a transformer-based approach, the pretrained:

```text
distilbert-base-uncased
```

model was fine-tuned using Google Colab with GPU acceleration.

The complaint narratives were tokenized using the corresponding DistilBERT tokenizer.

The model was configured for the same eight-class classification problem and used the same class mapping as the classical models.

The fine-tuned DistilBERT model achieved:

```text
Validation Accuracy:     85.09%
Validation Macro F1:     85.09%
Validation Weighted F1:  85.09%
```

The transformer therefore provided a stronger validation score than the classical models, but model selection was not based on validation score alone.

---

# 12. Model Comparison and Final Model Selection

The candidate models were compared using predictive performance together with measured deployment cost.

| Model | Validation Macro F1 | Model Size | RAM Usage | Inference Time |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.94% | 1.99 MB | 22.71 MB | 0.19 ms |
| XGBoost | 83.35% | 6.33 MB | 88.77 MB | 0.55 ms |
| DistilBERT | 85.09% | 256.12 MB | 703.36 MB | 0.82 ms |

DistilBERT achieved the highest validation Macro F1, exceeding XGBoost by **1.74 percentage points**.

However, the improvement came with substantially greater resource requirements. The measured DistilBERT model was approximately **40× larger than XGBoost** and used substantially more RAM during inference.

XGBoost provided a middle ground between the lightweight Logistic Regression baseline and the more resource-intensive transformer model.

Based on the combination of predictive performance, model size, memory requirements, and inference cost, **XGBoost was selected as the production model**.

## 12.1 Deployment Cost Benchmark

To compare the practical deployment cost of the candidate models, model size, RAM usage, and inference latency were measured locally.

The following benchmark was used for the XGBoost pipeline:

```python
import os, time, psutil, joblib

model_path = "models/xgb_pipeline.pkl"

# Model Size
model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

# RAM Usage
process = psutil.Process(os.getpid())

ram_before = process.memory_info().rss / (1024 ** 2)

# Load model
model = joblib.load(model_path)

ram_after = process.memory_info().rss / (1024 ** 2)
ram_usage_mb = ram_after - ram_before

# Inference Time
sample_text = X_test.iloc[0]

start = time.perf_counter()

prediction = model.predict([sample_text])

end = time.perf_counter()

inference_time_ms = (end - start) * 1000

print(f"Model size: {model_size_mb:.2f} MB")
print(f"RAM usage: {ram_usage_mb:.2f} MB")
print(f"Inference time: {inference_time_ms:.2f} ms")
```
---

# 13. Final Model Evaluation

After model selection, the XGBoost model was evaluated on the previously untouched test set.

The final test set contained:

```text
7,500 complaints
```

## Overall Test Results

| Metric | Result |
|---|---:|
| Accuracy | **83.23%** |
| Weighted Precision | **83.37%** |
| Recall | **83.23%** |
| Macro F1 | **83.26%** |
| Weighted F1 | **83.26%** |

The XGBoost validation accuracy was approximately **83.32%**, while the final test accuracy was **83.23%**.

The difference between the two accuracy measurements was approximately **0.09 percentage points**.

The final test evaluation was performed only after the model-selection stage so that the test set could provide an independent estimate of performance on previously unseen data.

---

# 14. Final Per-Class Performance

The final XGBoost test classification report was:

| Product Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Checking or savings account | 0.73 | 0.77 | **0.75** | 937 |
| Credit card | 0.84 | 0.81 | **0.83** | 937 |
| Credit reporting or other personal consumer reports | 0.84 | 0.92 | **0.88** | 938 |
| Debt collection | 0.86 | 0.84 | **0.85** | 938 |
| Money transfer, virtual currency, or money service | 0.78 | 0.78 | **0.78** | 938 |
| Mortgage | 0.95 | 0.91 | **0.93** | 937 |
| Payday loan, title loan, personal loan, or advance loan | 0.77 | 0.77 | **0.77** | 938 |
| Vehicle loan or lease | 0.89 | 0.87 | **0.88** | 937 |

Performance varied across categories despite the balanced dataset.

The highest F1 score was observed for **Mortgage (0.93)**, followed by **Credit reporting (0.88)** and **Vehicle loan or lease (0.88)**.

Lower F1 scores were observed for **Checking or savings account (0.75)**, **Payday loan/title/personal loan/advance loan (0.77)**, and **Money transfer (0.78)**.

---

# 15. Error Analysis

Detailed error analysis was performed using the XGBoost validation predictions.

Out of 7,500 validation samples:

```text
Validation samples: 7,500
Errors:             1,251
```

The analysis examined confusion pairs as well as individual misclassified complaints.

## 15.1 Major Confusion Patterns

Some of the most frequent confusion pairs were:

| Actual Category | Predicted Category | Errors |
|---|---|---:|
| Checking or savings account | Money transfer | 160 |
| Money transfer | Checking or savings account | 153 |
| Credit card | Checking or savings account | 67 |
| Vehicle loan or lease | Payday/title/personal loan | 59 |
| Mortgage | Payday/title/personal loan | 50 |
| Payday/title/personal loan | Debt collection | 49 |
| Checking or savings account | Credit card | 49 |
| Payday/title/personal loan | Vehicle loan | 48 |
| Debt collection | Credit reporting | 41 |
| Payday/title/personal loan | Credit reporting | 36 |

The largest source of confusion was between **checking/savings accounts and money-transfer services**.

---

# 16. Manual Error Inspection

Representative misclassified complaints were manually inspected rather than treating every incorrect prediction as the same type of model failure.

Several examples contained legitimate signals associated with more than one product category.

For example, a complaint could describe a transaction involving a bank account while simultaneously discussing a money-transfer service.

This creates ambiguity because the classifier is required to produce a single product category even when a complaint contains signals associated with multiple categories.

The error analysis therefore identified **product-category overlap and ambiguous complaint narratives** as important sources of classification difficulty.

---

# 17. Error Rate by Complaint Length

Validation errors were also analyzed according to complaint length.

| Complaint Length | Error Rate |
|---|---:|
| 1–25 words | **31.12%** |
| 26–50 words | **25.33%** |
| 51–100 words | **19.06%** |
| 101–200 words | **15.33%** |
| 201–500 words | **14.69%** |
| 500+ words | **10.71%** |

Shorter complaints had substantially higher observed validation error rates.

The error rate decreased as more textual information became available.

This is an observed relationship in the validation data and should not be interpreted as proof that complaint length directly causes classification errors.

---

# 18. MLflow Experiment Tracking

MLflow was used throughout the modeling stage to track experiments and their associated metrics and artifacts.

The tracking workflow included runs for the classical models and the transformer experiment.

Tracked information included:

### Parameters

- Model type
- TF-IDF configuration
- XGBoost hyperparameters
- Dataset split information
- Transformer configuration

### Metrics

- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1

### Artifacts

- Classification reports
- Confusion matrices
- Trained model artifacts
- Transformer model/tokenizer artifacts where applicable

MLflow provided a centralized record of the experiments and made it possible to compare the candidate approaches during model development.

---

# 19. Data and Model Versioning with DVC

DVC was used to separate large data and model artifacts from the Git source-code history.

The project configured a Google Drive DVC remote for artifact storage.

The relevant large files were tracked using DVC rather than committed directly to the Git repository.

This allowed the project to maintain reproducible references to datasets and model artifacts while keeping the Git repository focused on source code and documentation.

The DVC configuration uses local credentials for the remote rather than storing authentication information in Git.

---

# 20. Final Model Artifact

The finalized classical model was packaged into a single inference pipeline.

The pipeline contains:

```text
TF-IDF Vectorizer
        ↓
XGBoost Classifier
```

The primary inference artifact is:

```text
models/xgb_pipeline.pkl
```

The label encoder is stored separately:

```text
models/label_encoder.pkl
```

The production inference setup therefore requires:

```text
models/
├── xgb_pipeline.pkl
└── label_encoder.pkl
```

---

# 21. API Service and Deployment

## 21.1 Backend-Only Architecture

The project provides a **backend-only machine learning API service**. It does not include a dedicated frontend or web application interface.

Users and applications interact with the trained classifier through the FastAPI REST API. The API's interactive interface is available through FastAPI's automatically generated Swagger documentation at `/docs`.

The `/docs` interface allows users to inspect the available endpoints, provide complaint text, send prediction requests, and view the returned JSON response directly from a web browser.

The project architecture is therefore:

```text
Client / API Consumer
        ↓
FastAPI Backend
        ↓
ML Inference Pipeline
        ↓
XGBoost Prediction
        ↓
JSON Response
```

## 21.2 Prediction API

The finalized model was integrated into a **FastAPI** service.

The API exposes endpoints for health checking and complaint prediction.

The main prediction endpoint is:

```text
POST /predict
```

The inference flow is:

```text
Client
   ↓
FastAPI /predict
   ↓
Complaint Text
   ↓
xgb_pipeline.pkl
   ↓
TF-IDF
   ↓
XGBoost
   ↓
Encoded Prediction
   ↓
label_encoder.pkl
   ↓
Product Category
   ↓
JSON Response
```

The API also performs request validation and includes limits for prediction input sizes.

Prediction logging records information such as:

- Timestamp
- Endpoint
- Input length
- Predicted category
- Confidence
- Prediction latency

The service logs structured prediction information to the application output and also supports local file logging during local execution.

---

# 22. Containerization

The FastAPI service was containerized using Docker.

The production container includes the API source code, required runtime dependencies, and the finalized XGBoost model artifacts.

The runtime dependencies are maintained separately in:

```text
requirements-api.txt
```

The Docker image was published to **GitHub Container Registry (GHCR)**.

The published image uses the repository:

```text
ghcr.io/codeslavve/supportticket-complaintclassifier
```

Using a container provides a consistent runtime environment between local development and deployment.

---

# 23. CI/CD

GitHub Actions was used to automate the project's validation and container publishing workflow.

The CI/CD workflow runs when changes are pushed or pull requests are made against the main branch.

The workflow includes:

```text
Code Change
    ↓
Install Dependencies
    ↓
Run Tests
    ↓
Run Ruff
    ↓
Build Docker Image
    ↓
Publish Image to GHCR
```

This ensures that the project is tested and linted before the Docker image is published.

The resulting container image can then be used by the deployment environment.

---

# 24. Production Deployment

The FastAPI application is deployed as a Docker-based service on **Render**.

The deployed service exposes the complaint-classification API and provides an interactive Swagger/OpenAPI interface through:

```text
/docs
```

The production inference path is therefore:

```text
Client
   ↓
Render
   ↓
Docker Container
   ↓
FastAPI
   ↓
XGBoost Pipeline
   ↓
Prediction
```

The deployment provides a publicly accessible API for sending complaint narratives and receiving predicted CFPB product categories.

---

# 25. Known Limitations

## 25.1 Product Category Overlap

Several CFPB product categories have overlapping terminology and real-world situations.

The strongest observed confusion was between checking/savings accounts and money-transfer services.

Manual inspection of misclassified examples showed that some complaints contain signals associated with multiple products.

This creates an inherent difficulty in the classification task because the model is required to produce a single category even when the narrative contains evidence associated with multiple categories.

---

## 25.2 Short Complaints

Short complaints were significantly more difficult to classify.

The validation error rate was **31.12% for complaints containing 1–25 words**, compared with **10.71% for complaints containing more than 500 words**.

A short complaint may provide insufficient contextual information to distinguish between closely related financial products.

---

## 25.3 Uneven Category-Level Performance

Although the final dataset was balanced, the model did not achieve identical performance across all categories.

For example:

- Mortgage — **0.93 F1**
- Credit reporting — **0.88 F1**
- Vehicle loan — **0.88 F1**
- Checking/savings — **0.75 F1**

This shows that equal class representation does not necessarily result in equal classification difficulty.

The language used to describe different financial products can vary in distinctiveness and overlap.

---

## 25.4 TF-IDF Representation Limitations

The deployed model relies on TF-IDF features.

TF-IDF is effective at representing important words and n-grams, but it does not provide the same contextual representation as transformer-based models.

Complaints containing similar vocabulary across multiple product categories can therefore be difficult to separate.

---

## 25.5 Dataset Scope

The final dataset contains 50,000 sampled complaints from eight selected CFPB product categories.

The reported results therefore describe performance within this project dataset and selected label space.

They should not automatically be interpreted as performance across every CFPB complaint or every financial product category.

---

## 25.6 Deployment Resource Constraints

The production model was selected partly because of its lower resource requirements compared with the transformer alternative.

Although DistilBERT achieved higher validation performance, its measured memory requirement was substantially higher.

For this project, the additional predictive performance did not justify the additional deployment resource cost under the selected deployment constraints.

---

# 26. Future Improvements

Potential future improvements include:

- Increasing the size and diversity of the training dataset.
- Investigating additional CFPB product categories.
- Improving handling of ambiguous or overlapping labels.
- Exploring more advanced text representations.
- Further fine-tuning and optimization of transformer-based models.
- Evaluating confidence scores for predictions.
- Introducing human review for low-confidence or ambiguous complaints.
- Monitoring model performance after deployment.
- Periodically retraining the model as new complaint data becomes available.

---

# 27. Project Structure

The repository is organized around data preparation, experimentation, model artifacts, API deployment, testing, and documentation.

```text
SupportTicket-ComplaintClassifier/
│
├── .dvc/
│
├── data/
│   └── cfpb_sample_50k.csv
│
├── docs/
│
├── mlartifacts/
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
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── xgb_confusion_pairs.csv
│   └── xgb_validation_errors.csv
│
├── src/
│   ├── api/
│   │   ├── logger.py
│   │   ├── main.py
│   │   ├── model.py
│   │   └── schemas.py
│   │
│   ├── scripts/
│   │   └── sample_cfpb.py
│   │
│   └── __init__.py
│
├── tests/
│
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
├── project_writeup.md
└── ...
```

Large datasets, trained model files, experiment artifacts, and other generated files are kept out of normal Git tracking where appropriate and are managed through DVC or ignored according to the repository configuration.

---

# 28. End-to-End Project Workflow

The complete project workflow can be summarized as:

```text
CFPB Complaint Data
        ↓
Data Sampling
        ↓
Data Cleaning & Category Selection
        ↓
Balanced 50,000-Complaint Dataset
        ↓
Train / Validation / Test Split
        ↓
TF-IDF Representation
        ↓
Logistic Regression
        ↓
XGBoost
        ↓
DistilBERT Fine-Tuning
        ↓
Model Comparison
        ↓
XGBoost Selection
        ↓
Validation Error Analysis
        ↓
Final Test Evaluation
        ↓
MLflow Experiment Tracking
        ↓
DVC Data / Artifact Versioning
        ↓
FastAPI Inference Service
        ↓
Docker Container
        ↓
GitHub Actions CI/CD
        ↓
GHCR
        ↓
Render Deployment
```

---

# 29. Conclusion

This project developed an end-to-end text classification workflow for categorizing CFPB consumer complaints into eight financial product categories.

The workflow progressed from sampled CFPB complaint data through data preparation, classical NLP, transformer experimentation, model comparison, error analysis, final evaluation, experiment tracking, artifact versioning, API development, containerization, CI/CD, and deployment.

The final production model was **XGBoost with TF-IDF features**.

On the untouched test set, the model achieved:

```text
Accuracy:  83.23%
Macro F1:  83.26%
```

DistilBERT achieved a higher validation Macro F1 of **85.09%**, but required substantially more memory and model storage than XGBoost.

XGBoost was therefore selected as the production model based on the combined consideration of predictive performance and deployment cost.

The final inference artifacts are:

```text
models/xgb_pipeline.pkl
models/label_encoder.pkl
```

The model is integrated into a FastAPI service, containerized using Docker, published through GitHub Container Registry, and deployed on Render.

Error analysis identified **overlapping product categories** and **short complaint narratives** as important sources of classification difficulty.

The project therefore demonstrates a complete machine learning workflow extending beyond model training into evaluation, experiment tracking, reproducibility, API development, containerization, CI/CD, and deployment.
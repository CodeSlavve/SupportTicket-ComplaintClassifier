# Model Card — CFPB Complaint Classifier

## 1. Model Overview

This model is a multi-class text classification model trained to categorize consumer complaint narratives into eight CFPB product categories.

The production model is an **XGBoost classifier** using a **TF-IDF text representation**. The complete preprocessing and classification pipeline is stored in:

```text
models/xgb_pipeline.pkl
```

The corresponding label encoder is stored separately:

```text
models/label_encoder.pkl
```

The model was selected after comparing Logistic Regression, XGBoost, and DistilBERT models using validation performance and resource requirements.

---

## 2. Model Details

| Property                   | Details                               |
| -------------------------- | ------------------------------------- |
| Task                       | Multi-class text classification       |
| Input                      | Consumer complaint narrative          |
| Output                     | One of 8 complaint product categories |
| Production model           | XGBoost                               |
| Text representation        | TF-IDF                                |
| TF-IDF features            | 20,000                                |
| N-grams                    | Unigrams + bigrams                    |
| Minimum document frequency | 5                                     |
| Training data              | 35,000 complaints                     |
| Validation data            | 7,500 complaints                      |
| Test data                  | 7,500 complaints                      |
| Total dataset              | 50,000 complaints                     |

The final dataset contains eight balanced product classes with **6,250 complaints per class**.

---

## 3. Output Classes

The model predicts one of the following eight categories:

1. Checking or savings account
2. Credit card
3. Credit reporting or other personal consumer reports
4. Debt collection
5. Money transfer, virtual currency, or money service
6. Mortgage
7. Payday loan, title loan, personal loan, or advance loan
8. Vehicle loan or lease

---

## 4. Capabilities

The model can:

- Classify a consumer complaint narrative into one of eight product categories.
- Process individual complaint text through the deployed FastAPI API.
- Return the predicted category and model confidence.
- Process batch prediction requests through the API.
- Operate as a relatively small CPU-based text classification model suitable for a lightweight API deployment.

The production pipeline contains the TF-IDF transformation and XGBoost classifier, so the API does not require a separate text-vectorization step.

---

## 5. Intended Use

The model is intended for:

- Demonstrating automated complaint categorization.
- Supporting experimentation with classical NLP classification.
- Providing a portfolio-scale example of an end-to-end ML deployment.
- Categorizing CFPB-style consumer complaint narratives into broad product categories.

The model is not intended to make financial decisions, determine legal outcomes, resolve complaints automatically, or replace human review.

---

## 6. Dataset Scope

The model was developed using a **50,000-complaint sample of CFPB consumer complaint data**.

The working dataset contains complaint-related fields, with the modeling task using:

```text
clean_complaint_text
Product
```

The data was balanced across eight selected product categories, with 6,250 examples per category.

The dataset was split into:

- Training: 35,000
- Validation: 7,500
- Test: 7,500

The test set was kept separate until final model evaluation.

The dataset represents the selected sample and categories used in this project. Therefore, the reported performance should not be interpreted as representative of all CFPB complaints or all possible financial complaint categories.

---

## 7. Performance

### Final XGBoost Test Performance

| Metric          |  Score |
| --------------- | -----: |
| Accuracy        | 83.23% |
| Macro Precision | 83.37% |
| Macro Recall    | 83.23% |
| Macro F1        | 83.26% |
| Weighted F1     | 83.26% |

### Per-Class F1 Scores

| Complaint Category                                      | F1 Score |
| ------------------------------------------------------- | -------: |
| Checking or savings account                             |      75% |
| Credit card                                             |      83% |
| Credit reporting or other personal consumer reports     |      88% |
| Debt collection                                         |      85% |
| Money transfer, virtual currency, or money service      |      78% |
| Mortgage                                                |      93% |
| Payday loan, title loan, personal loan, or advance loan |      77% |
| Vehicle loan or lease                                   |      88% |

---

## 8. Model Comparison

The production model was selected after comparing three approaches on the validation set.

| Model               | Validation Macro F1 | Model Size | RAM Usage | Inference |
| ------------------- | ------------------: | ---------: | --------: | --------: |
| Logistic Regression |              82.94% |    1.99 MB |  22.71 MB |   0.19 ms |
| XGBoost             |              83.35% |    6.33 MB |  88.77 MB |   0.55 ms |
| DistilBERT          |              85.09% |  256.12 MB | 703.36 MB |   0.82 ms |

DistilBERT achieved a **1.74 percentage-point higher validation Macro F1** than XGBoost, but required substantially more memory.

XGBoost was therefore selected for production based on the overall performance and resource tradeoff rather than validation F1 alone.

---

## 9. Error Patterns

Error analysis showed that classification performance varies with complaint length.

| Complaint Length   | Error Rate |
| ------------------ | ---------: |
| 1–25 characters    |     31.12% |
| 26–50 characters   |     25.33% |
| 51–100 characters  |     19.06% |
| 101–200 characters |     15.33% |
| 201–500 characters |     14.69% |
| 500+ characters    |     10.71% |

Short complaints produced higher error rates, while longer narratives generally provided more information for classification.

The largest validation confusion was between:

- **Checking or savings account → Money transfer:** 160 cases
- **Money transfer → Checking or savings account:** 153 cases

This indicates that complaints involving account activity and money transfers can contain overlapping terminology and context.

---

## 10. Limitations

The model has several limitations:

- It only predicts the eight product categories used during training.
- It cannot classify categories that were excluded from the project dataset.
- Performance depends on the quality and content of the complaint narrative.
- Short or vague complaints are more difficult to classify.
- Some product categories contain overlapping terminology, which can lead to confusion.
- The reported metrics are based on a 50,000-complaint sample and may not generalize to other datasets or future complaint distributions.
- The confidence score should not be interpreted as a guaranteed probability that the prediction is correct.
- The model performs text classification only and does not understand the full legal, financial, or factual context of a complaint.

---

## 11. Production Deployment

The model is deployed through a FastAPI service.

The production pipeline is:

```text
Complaint Text
      ↓
FastAPI API
      ↓
TF-IDF Transformation
      ↓
XGBoost Classifier
      ↓
Predicted Category + Confidence
```

The API exposes:

```text
GET  /health
POST /predict
POST /predict/batch
```

Prediction requests are logged with operational metadata including timestamp, endpoint, input length, predicted category, confidence, and latency.

The model is packaged into the Docker image used for deployment.

---

## 12. Model Artifacts

The production artifacts are:

```text
models/xgb_pipeline.pkl
models/label_encoder.pkl
```

`xgb_pipeline.pkl` contains the TF-IDF transformation and XGBoost classifier.

`label_encoder.pkl` maps the model's encoded class labels back to the original complaint categories.

The artifacts are versioned using DVC rather than being stored directly in Git.

---

## 13. Reproducibility

The project tracks the production dataset and model artifacts with DVC.

The DVC remote is configured using Google Drive, while sensitive local credentials are kept outside version control.

The project also includes:

- Training and evaluation notebooks
- MLflow experiment tracking
- FastAPI API implementation
- Automated tests
- Docker configuration
- GitHub Actions CI/CD
- Container publishing through GHCR
- Render deployment

---

## 14. Model Status

**Production model:** XGBoost + TF-IDF

**Validation Macro F1:** 83.35%

**Final Test Macro F1:** 83.26%

**Deployment:** FastAPI + Docker + Render

**Primary purpose:** Multi-class classification of selected CFPB consumer complaint narratives.
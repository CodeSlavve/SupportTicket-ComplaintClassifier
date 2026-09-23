# Day 4 — Model Selection + Error Analysis

## Objective

The objective of Day 4 was to select the final production model from the three approaches evaluated during the project:

* Logistic Regression
* XGBoost
* Fine-tuned DistilBERT

The decision was based primarily on **Macro-F1**, while also considering model size, memory usage, inference performance, and deployment practicality.

After selecting the production model, its validation errors were analyzed to understand recurring failure patterns. Only after the model was finalized was the previously untouched test set used for the final evaluation.

---

# 1. Model Comparison

The three models were compared using their validation Macro-F1 scores and operational characteristics.

| Model               | Validation Macro-F1 |  Model Size | Measured RAM | Inference Benchmark |
| ------------------- | ------------------: | ----------: | -----------: | ------------------: |
| Logistic Regression |              82.94% |     1.99 MB |     22.71 MB |             0.19 ms |
| **XGBoost**         |          **83.35%** | **6.33 MB** | **88.77 MB** |         **0.55 ms** |
| DistilBERT          |          **85.09%** |   256.12 MB |    703.36 MB |             0.82 ms |

DistilBERT achieved the highest validation Macro-F1 at **85.09%**.

However, its improvement over XGBoost was relatively small:

```text
DistilBERT: 85.09%
XGBoost:    83.35%

Difference: 1.74 percentage points
```

The performance improvement therefore had to be considered alongside the additional operational requirements of the transformer model.

---

# 2. Production Model Selection

## Selected Model: XGBoost

XGBoost was selected as the official production model.

The decision was based on the tradeoff between predictive performance and deployment cost rather than simply selecting the model with the highest Macro-F1.

Compared with XGBoost, DistilBERT:

* had a model size approximately **40.5× larger**
* used approximately **7.9× more measured RAM**
* required substantially more computational resources for CPU processing
* required substantially more computational resources for CPU-based inference and had higher measured memory usage

XGBoost achieved **83.35% validation Macro-F1** while requiring substantially fewer resources and providing a simpler inference pipeline.

XGBoost also improved over the Logistic Regression baseline:

```text
Logistic Regression: 82.94%
XGBoost:             83.35%

Improvement:         +0.41 percentage points
```

Therefore, XGBoost provided a practical middle ground between the very lightweight Logistic Regression model and the more resource-intensive DistilBERT model.

The final decision was:

> **XGBoost was selected as the production model because it provided a strong validation Macro-F1 of 83.35% while maintaining substantially lower storage and memory requirements than DistilBERT. Although DistilBERT achieved a higher Macro-F1 of 85.09%, its significantly higher resource requirements and slower CPU-based processing made XGBoost a more practical choice for the intended deployment.**

DistilBERT was therefore retained as an experimental benchmark, while XGBoost became the model used for final evaluation and deployment.

---

# 3. XGBoost Error Analysis

After selecting XGBoost, its validation predictions were analyzed to understand where the model was making mistakes.

The validation set contained:

```text
Validation samples: 7,500
Incorrect predictions: 1,251
Validation accuracy: ~83.32%
```

The analysis included:

* confusion matrix inspection;
* identifying the most frequent confusion pairs;
* manually inspecting representative misclassified complaints;
* analyzing error rates by product category;
* analyzing error rates by complaint length.

---

## 3.1 Major Confusion Pairs

The most frequent validation confusions included:

| Actual Category             | Predicted Category          | Errors |
| --------------------------- | --------------------------- | -----: |
| Checking or savings account | Money transfer              |    160 |
| Money transfer              | Checking or savings account |    153 |
| Credit card                 | Checking or savings account |     67 |
| Vehicle loan or lease       | Payday/title/personal loan  |     59 |
| Mortgage                    | Payday/title/personal loan  |     50 |
| Payday/title/personal loan  | Debt collection             |     49 |
| Checking or savings account | Credit card                 |     49 |
| Payday/title/personal loan  | Vehicle loan                |     48 |
| Debt collection             | Credit reporting            |     41 |
| Payday/title/personal loan  | Credit reporting            |     36 |

The strongest confusion occurred between:

```text
Checking or savings account
            ↕
Money transfer, virtual currency, or money service
```

These two categories accounted for a substantial number of errors in both directions.

---

# 4. Manual Inspection of Misclassified Complaints

Representative validation errors were manually inspected to determine whether the model was clearly incorrect or whether the complaint itself contained ambiguous signals.

Several complaints contained information associated with more than one product category.

For example, a complaint could involve a bank account while describing a transaction or transfer service at the same time.

This means that some errors cannot be attributed entirely to the classifier.

In certain cases, the complaint narrative itself provides evidence for multiple categories while the dataset requires a single target label.

The error analysis therefore identified **category overlap and label ambiguity** as an important source of difficulty.

This is relevant because such ambiguity represents a limitation of the classification task and dataset, rather than something that can necessarily be solved by changing the classifier alone.

---

# 5. Failure Patterns by Product Category

The validation error rate was calculated separately for each actual product category.

| Product Category                                        | Total | Errors | Error Rate |
| ------------------------------------------------------- | ----: | -----: | ---------: |
| Checking or savings account                             |   938 |    250 | **26.65%** |
| Payday loan, title loan, personal loan, or advance loan |   937 |    213 | **22.73%** |
| Money transfer, virtual currency, or money service      |   937 |    199 | **21.24%** |
| Credit card                                             |   938 |    176 | **18.76%** |
| Vehicle loan or lease                                   |   938 |    137 | **14.61%** |
| Debt collection                                         |   937 |    130 | **13.87%** |
| Mortgage                                                |   938 |     76 |  **8.10%** |
| Credit reporting or other personal consumer reports     |   937 |     70 |  **7.47%** |

The model's errors were not distributed equally across categories.

Checking or savings account complaints had the highest validation error rate at **26.65%**, followed by payday/title/personal/advance loan complaints at **22.73%**.

Mortgage and credit reporting complaints had substantially lower observed validation error rates.

Because the dataset was perfectly balanced, these differences were not caused by one class having substantially fewer training examples. They instead indicate differences in classification difficulty and overlap between categories.

---

# 6. Failure Patterns by Complaint Length

Complaint length was also examined to determine whether the amount of available text was associated with classification errors.

| Complaint Length | Total | Errors | Error Rate |
| ---------------- | ----: | -----: | ---------: |
| 1–25 words       |   286 |     89 | **31.12%** |
| 26–50 words      |   537 |    136 | **25.33%** |
| 51–100 words     | 1,238 |    236 | **19.06%** |
| 101–200 words    | 2,251 |    345 | **15.33%** |
| 201–500 words    | 2,608 |    383 | **14.69%** |
| 500+ words       |   579 |     62 | **10.71%** |

The shortest complaints had the highest observed error rate.

Complaints containing **1–25 words had a 31.12% error rate**, compared with **10.71% for complaints containing more than 500 words**.

This suggests that complaints containing less textual information were more difficult for the classifier to distinguish.

This is an observed relationship in the validation data and does not establish that complaint length itself causes classification errors.

---

# 7. Final Test Evaluation

Once XGBoost had been selected and the validation error analysis was completed, the previously untouched test set was evaluated.

The test set contained:

```text
7,500 complaints
```

The test set was not used during model selection or error analysis.

The final XGBoost test results were:

| Metric             | Test Result |
| ------------------ | ----------: |
| Accuracy           |  **83.23%** |
| Weighted Precision |  **83.37%** |
| Recall             |  **83.23%** |
| Macro F1           |  **83.26%** |
| Weighted F1        |  **83.26%** |

The validation accuracy was approximately **83.32%**, while the final test accuracy was **83.23%**.

The difference was approximately:

```text
83.32% - 83.23% = 0.09 percentage points
```

The close validation and test results indicate that the model maintained similar performance on previously unseen data.

The final test result became the official reported performance of the production model.

---

# 8. Final Deployment Artifacts

After the final model evaluation, the artifacts required for inference were finalized.

The production `models/` directory contains:

```text
models/
├── xgb_pipeline.pkl
└── label_encoder.pkl
```

### `xgb_pipeline.pkl`

Contains the complete classical inference pipeline:

```text
Input Complaint
      ↓
TF-IDF Vectorizer
      ↓
XGBoost Classifier
      ↓
Encoded Prediction
```

Because the TF-IDF vectorizer is already contained within the pipeline, a separate TF-IDF file is not required by the API.

### `label_encoder.pkl`

Converts the numeric prediction produced by the classifier back into the original CFPB product category.

The finalized model artifacts therefore match the files that will be loaded by the FastAPI service.

---

# 9. Day 4 Outcome

By the end of Day 4, the project had:

* compared Logistic Regression, XGBoost, and DistilBERT;
* evaluated the models using validation Macro-F1;
* selected **XGBoost as the production model** based on the performance/resource tradeoff;
* identified the major XGBoost confusion pairs;
* manually inspected validation errors;
* identified category overlap as an important source of ambiguity;
* identified higher error rates among shorter complaints;
* evaluated category-specific failure rates;
* locked the production model before accessing the test set;
* evaluated XGBoost once on the untouched test set;
* recorded the final **83.26% Macro-F1** and **83.23% accuracy**;
* finalized the model artifacts required for deployment.

The model-selection stage was therefore completed before moving to the deployment stage.

---

# 10. Transition to Day 5

With the production model locked and its artifacts finalized, the project moved from model development into deployment.

The next stage was to build the FastAPI service around:

```text
models/xgb_pipeline.pkl
models/label_encoder.pkl
```

The API would expose the trained classifier through a `/predict` endpoint that accepts complaint text and returns the predicted product category.

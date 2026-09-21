# CFPB Consumer Complaint Classifier

## 1. Project Overview

The **CFPB Consumer Complaint Classifier** is a multi-class text classification system designed to automatically categorize consumer financial complaints into their corresponding product categories.

The project uses complaint narratives from the **Consumer Financial Protection Bureau (CFPB)** and applies natural language processing and machine learning techniques to classify complaints.

Two different approaches were explored:

1. **Classical NLP:** TF-IDF + machine learning
2. **Transformer NLP:** Fine-tuned DistilBERT

The classical approach evaluated Logistic Regression and XGBoost, while DistilBERT was fine-tuned as the transformer-based approach.

The project ultimately focused on evaluating these approaches using consistent validation and final test-set evaluation, followed by error analysis and preparation of the selected model for API deployment.

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

* Prepare a clean dataset from CFPB consumer complaint data.
* Select a focused set of financial product categories.
* Build a balanced dataset for classification.
* Establish a classical NLP baseline using TF-IDF.
* Compare Logistic Regression and XGBoost.
* Fine-tune a pretrained DistilBERT model.
* Compare classical NLP with transformer-based NLP.
* Evaluate the selected model on an untouched test set.
* Analyze model errors and identify recurring failure patterns.
* Track experiments and artifacts using MLflow.
* Prepare the finalized model artifacts for inference through a FastAPI service.

---

# 4. Dataset

The source data was obtained from the **Consumer Financial Protection Bureau consumer complaint database**.

The original CFPB dataset is very large, making it impractical and unnecessary to process the complete dataset for this project.

A representative working dataset was therefore created by sampling relevant complaints from the available CFPB data.

After filtering, cleaning, and category selection, the final dataset contained:

```text
Rows:        50,000
Columns:     2
Classes:     8
```

The final dataset contains only the two fields required for the classification task:

```text
cleaned_complaint_text
Products
```

Where:

* `cleaned_complaint_text` is the model input.
* `Products` is the target label.

---

# 5. Selected Product Categories

Eight product categories were finalized for the classification task:

1. Mortgage
2. Debt collection
3. Credit reporting or other personal consumer reports
4. Vehicle loan or lease
5. Payday loan, title loan, personal loan, or advance loan
6. Credit card
7. Money transfer, virtual currency, or money service
8. Checking or savings account

The dataset was balanced so that each category contained exactly **6,250 complaints**.

| Product Category                                        |    Samples |
| ------------------------------------------------------- | ---------: |
| Mortgage                                                |      6,250 |
| Debt collection                                         |      6,250 |
| Credit reporting or other personal consumer reports     |      6,250 |
| Vehicle loan or lease                                   |      6,250 |
| Payday loan, title loan, personal loan, or advance loan |      6,250 |
| Credit card                                             |      6,250 |
| Money transfer, virtual currency, or money service      |      6,250 |
| Checking or savings account                             |      6,250 |
| **Total**                                               | **50,000** |

This resulted in an equal **12.5% representation for each class**.

---

# 6. Data Preprocessing

The original CFPB complaint data contained many fields that were not required for the text classification task.

After the necessary filtering and text preprocessing, the irrelevant original columns were removed.

The final dataset was reduced to:

```text
cleaned_complaint_text
Products
```

Rows without usable complaint narratives were excluded because a text classifier cannot make a prediction without complaint text.

The text preprocessing produced the `cleaned_complaint_text` field used by the downstream NLP pipeline.

The target labels were retained in `Products`.

---

# 7. Dataset Split

The 50,000 samples were divided into training, validation, and test sets.

| Split      |    Samples | Percentage |
| ---------- | ---------: | ---------: |
| Training   |     35,000 |        70% |
| Validation |      7,500 |        15% |
| Test       |      7,500 |        15% |
| **Total**  | **50,000** |   **100%** |

The split was maintained consistently across the classical NLP and transformer experiments.

The test set was kept untouched during model development and was used only for the final evaluation of the selected model.

---

# 8. Approach 1 — Classical NLP

## 8.1 TF-IDF

The first modeling approach used **Term Frequency–Inverse Document Frequency (TF-IDF)** to convert complaint narratives into numerical feature vectors.

The vectorizer was fitted only on the training data and then used to transform the validation and test sets.

The configuration included:

```text
max_features = 20,000
ngram_range = (1, 2)
```

The use of `(1, 2)` allowed the representation to include both individual words and two-word combinations.

The resulting training matrix had the shape:

```text
(35,000, 20,000)
```

---

# 9. Logistic Regression

Logistic Regression was used as the first classical machine learning baseline.

It was trained on the TF-IDF representation and evaluated on the validation set.

The model provided a computationally efficient baseline for the multi-class text classification problem.

---

# 10. XGBoost

XGBoost was used as the second classical model.

The model operated on the same TF-IDF representation used by Logistic Regression.

The configuration used for the XGBoost experiment was:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.1
random_state = 42
n_jobs = -1
```

The use of the same dataset split and TF-IDF representation allowed XGBoost to be compared directly with Logistic Regression.

XGBoost was selected as the classical baseline to carry forward to the transformer comparison.

---

# 11. Approach 2 — Fine-Tuned DistilBERT

To evaluate a transformer-based approach, the pretrained:

```text
distilbert-base-uncased
```

model was fine-tuned using Google Colab with GPU acceleration.

The complaint narratives were tokenized using the corresponding DistilBERT tokenizer.

The model was configured for the same eight-class classification problem.

The same label mapping used by the classical models was maintained for the transformer experiment.

The goal was to determine whether contextual representations learned by a pretrained transformer could improve upon the TF-IDF + XGBoost baseline.

---

# 12. Model Comparison

The project compared the classical and transformer approaches using the validation set.

The primary comparison metric was **Macro F1**, supported by accuracy and weighted F1.

The XGBoost validation accuracy was approximately:

```text
83.32%
```

The fine-tuned DistilBERT model achieved:

```text
Accuracy:     85.00%
Macro F1:     85.00%
Weighted F1:  85.00%
```

This represented an improvement of approximately **1.68 percentage points in validation accuracy** for DistilBERT over the XGBoost validation accuracy.

The validation results established DistilBERT as a stronger transformer benchmark, while the final model decision was based on the broader evaluation and deployment considerations rather than validation accuracy alone.

---

# 13. Final Model Evaluation

After the model development stage was completed, the selected XGBoost model was evaluated once on the previously untouched test set.

The final test set contained:

```text
7,500 complaints
```

## Overall Test Results

| Metric             |     Result |
| ------------------ | ---------: |
| Accuracy           | **83.23%** |
| Weighted Precision | **83.37%** |
| Recall             | **83.23%** |
| Macro F1           | **83.26%** |
| Weighted F1        | **83.26%** |

The validation accuracy of XGBoost was approximately **83.32%**, while its final test accuracy was **83.23%**.

The difference between validation and test accuracy was only approximately **0.09 percentage points**, indicating consistent performance between the validation and previously unseen test data.

---

# 14. Final Per-Class Performance

The final XGBoost test classification report was:

| Product Category                                        | Precision | Recall |       F1 | Support |
| ------------------------------------------------------- | --------: | -----: | -------: | ------: |
| Checking or savings account                             |      0.73 |   0.77 | **0.75** |     937 |
| Credit card                                             |      0.84 |   0.81 | **0.83** |     937 |
| Credit reporting or other personal consumer reports     |      0.84 |   0.92 | **0.88** |     938 |
| Debt collection                                         |      0.86 |   0.84 | **0.85** |     938 |
| Money transfer, virtual currency, or money service      |      0.78 |   0.78 | **0.78** |     938 |
| Mortgage                                                |      0.95 |   0.91 | **0.93** |     937 |
| Payday loan, title loan, personal loan, or advance loan |      0.77 |   0.77 | **0.77** |     938 |
| Vehicle loan or lease                                   |      0.89 |   0.87 | **0.88** |     937 |

The model performed particularly strongly on mortgage, credit reporting, vehicle loans, and debt collection.

Lower F1 scores were observed for checking/savings accounts, payday/title/personal/advance loans, and money-transfer complaints.

---

# 15. Error Analysis

Detailed error analysis was performed on the XGBoost validation predictions.

Out of 7,500 validation samples:

```text
Errors: 1,251
Validation accuracy: ~83.32%
```

The analysis examined both the most frequent confusion pairs and individual misclassified complaints.

## 15.1 Major Confusion Patterns

Some of the most frequent confusion pairs included:

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

The largest source of confusion was between **checking/savings accounts and money-transfer services**.

---

# 16. Manual Error Inspection

Representative misclassified complaints were manually inspected rather than treating every incorrect prediction as an identical model failure.

Several examples contained legitimate signals associated with more than one product category.

For example, a complaint could describe a transaction involving a bank account while simultaneously discussing a money-transfer service.

This makes some cases inherently difficult to categorize from the narrative alone.

Therefore, a portion of the observed error appears to be related to **overlap and ambiguity between the underlying product categories**, rather than solely to limitations of the classifier.

---

# 17. Error Rate by Complaint Length

Validation errors were also analyzed according to complaint length.

| Complaint Length | Error Rate |
| ---------------- | ---------: |
| 1–25 words       | **31.12%** |
| 26–50 words      | **25.33%** |
| 51–100 words     | **19.06%** |
| 101–200 words    | **15.33%** |
| 201–500 words    | **14.69%** |
| 500+ words       | **10.71%** |

Shorter complaints showed substantially higher validation error rates.

The error rate decreased as more textual information became available.

This is an observed relationship in the validation data and should not be interpreted as proof that complaint length directly causes classification errors.

---

# 18. MLflow Experiment Tracking

MLflow was used to track the machine learning experiments throughout the project.

The tracking workflow included separate experiments/runs for the classical models and transformer experiment.

Tracked information included:

### Parameters

* Model type
* TF-IDF configuration
* XGBoost hyperparameters
* Dataset split information
* Transformer configuration

### Metrics

* Accuracy
* Precision
* Recall
* Macro F1
* Weighted F1

### Artifacts

* Classification reports
* Confusion matrices
* Trained model artifacts
* Tokenizer/model artifacts where applicable

MLflow provided a centralized record of the experiments and made it possible to compare different modeling approaches.

---

# 19. Final Model Artifact

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

Therefore, the finalized XGBoost inference setup requires:

```text
models/
├── xgb_pipeline.pkl
└── label_encoder.pkl
```

The separately saved TF-IDF vectorizer is not required by the FastAPI service because the TF-IDF transformation is already contained inside `xgb_pipeline.pkl`.

---

# 20. Deployment Plan

The finalized model is intended to be exposed through a **FastAPI** service.

The inference flow will be:

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

The API will accept a complaint narrative and return the predicted CFPB product category.

---

# 21. Known Limitations

## 21.1 Product Category Overlap

Several CFPB product categories have overlapping terminology and real-world situations.

The strongest observed confusion was between checking/savings accounts and money-transfer services.

Manual inspection of misclassified examples showed that some complaints could reasonably be associated with multiple categories.

This creates an inherent difficulty in the classification task because the model is required to produce a single category even when the narrative contains signals associated with multiple products.

---

## 21.2 Short Complaints

Short complaints were significantly more difficult to classify.

The validation error rate was **31.12% for complaints containing 1–25 words**, compared with **10.71% for complaints containing more than 500 words**.

A short complaint may not provide enough contextual information to distinguish between closely related financial products.

---

## 21.3 Uneven Category-Level Performance

Although the final dataset was balanced, the model did not achieve identical performance across all categories.

For example:

* Mortgage — **0.93 F1**
* Credit reporting — **0.88 F1**
* Vehicle loan — **0.88 F1**
* Checking/savings — **0.75 F1**

This indicates that having equal numbers of training samples does not necessarily result in equal classification difficulty.

The language used to describe different financial products varies in distinctiveness and overlap.

---

## 21.4 TF-IDF Representation Limitations

The deployed classical model relies on TF-IDF features.

TF-IDF is effective at capturing important words and n-grams, but it does not provide the same contextual representation as transformer-based models.

Complaints containing similar vocabulary across multiple product categories can therefore be difficult to separate.

---

## 21.5 Dataset Scope

The final dataset contains 50,000 sampled complaints from eight selected CFPB product categories.

The results therefore describe performance within this project dataset and selected label space.

They should not automatically be interpreted as performance across every CFPB complaint or every financial product category.

---

# 22. Future Improvements

Potential future improvements include:

* Increasing the size and diversity of the training dataset.
* Investigating additional CFPB product categories.
* Improving handling of ambiguous or overlapping labels.
* Exploring more advanced text representations.
* Further fine-tuning and optimization of transformer-based models.
* Evaluating confidence scores for predictions.
* Introducing human review for low-confidence or ambiguous complaints.
* Monitoring model performance after deployment.
* Periodically retraining the model as new complaint data becomes available.

---

# 23. Project Structure

The project is organized around data, models, results, experimentation, and deployment:

```text
SupportTicket-ComplaintClassifier/
│
├── data/
│   └── cfpb_sample_50k.csv
│
├── models/
│   ├── xgb_pipeline.pkl
│   └── label_encoder.pkl
│
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── xgb_validation_errors.csv
│   └── xgb_confusion_pairs.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── src/
│   ├── predict.py
│   └── api.py
│
├── requirements.txt
├── README.md
└── project_writeup.md
```

Files may be excluded from version control when they are large or contain local experiment state.

---

# 24. Conclusion

This project developed an end-to-end text classification workflow for categorizing CFPB consumer complaints into eight financial product categories.

The workflow progressed from raw complaint data through:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Category Selection
      ↓
Balanced Dataset
      ↓
Train / Validation / Test Split
      ↓
TF-IDF
      ↓
Classical ML Baselines
      ↓
DistilBERT Fine-Tuning
      ↓
Model Evaluation
      ↓
Error Analysis
      ↓
Final Model
      ↓
Inference Pipeline
      ↓
FastAPI Deployment
```

The classical approach using TF-IDF and XGBoost achieved **83.23% accuracy and 83.26% Macro F1 on the untouched test set**.

The fine-tuned DistilBERT model achieved **85.00% validation accuracy and 85.00% validation Macro F1**, providing a useful transformer benchmark against the classical approach.

Detailed error analysis showed that the primary challenges were overlapping product categories and short complaint narratives.

The finalized XGBoost inference pipeline was packaged into `xgb_pipeline.pkl`, with `label_encoder.pkl` providing the mapping back to human-readable product categories.

The project therefore provides not only a trained classifier, but a complete workflow covering dataset preparation, classical NLP, transformer experimentation, evaluation, experiment tracking, error analysis, and preparation for API-based deployment.

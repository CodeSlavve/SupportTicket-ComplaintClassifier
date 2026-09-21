# Day 2 — Classical NLP Baseline + MLflow Tracking

## Objective

The objective of Day 2 was to establish a strong classical NLP baseline for the CFPB complaint classification task.

The cleaned complaint text from Day 1 was converted into numerical features using TF-IDF, and two classical machine learning models were trained and evaluated:

* Logistic Regression
* XGBoost

The selected baseline would later serve as the comparison point for the fine-tuned DistilBERT model.

MLflow was also introduced to track the experiments, parameters, and evaluation metrics in a reproducible way.

---

## 1. Dataset

The finalized dataset from Day 1 contained:

* **50,000 complaints**
* **8 product categories**
* **6,250 samples per category**
* `cleaned_complaint_text` — input text
* `Products` — target label

The dataset was split into three subsets:

| Split      |    Samples |
| ---------- | ---------: |
| Training   |     35,000 |
| Validation |      7,500 |
| Test       |      7,500 |
| **Total**  | **50,000** |

The validation set was used for model comparison during development.

The test set was kept untouched and reserved for the final evaluation after the model was finalized.

---

## 2. Label Encoding

The `Products` target column contains categorical text labels.

These labels were converted into numerical class IDs using a `LabelEncoder` so that the machine learning models could work with the target values.

The encoder was fitted on the training labels and retained so that predictions could later be converted back into the original product category names.

The resulting label encoder was eventually saved as:

```text
models/label_encoder.pkl
```

---

## 3. TF-IDF Vectorization

Since traditional machine learning algorithms cannot directly process raw text, the complaint narratives were converted into numerical feature vectors using `TfidfVectorizer`.

The vectorizer was fitted **only on the training data**.

The fitted vectorizer was then used to transform the validation and test sets.

This prevented information from the validation and test sets from leaking into the feature extraction process.

### TF-IDF configuration

```text
max_features = 20,000
ngram_range = (1, 2)
```

The `(1, 2)` n-gram configuration allowed the model to learn from both individual words and two-word combinations.

For example:

```text
credit
card
credit card
```

could all become separate TF-IDF features.

The resulting training matrix was:

```text
(35,000, 20,000)
```

This means the 35,000 training complaints were represented using up to 20,000 TF-IDF features.

---

## 4. Logistic Regression

The first baseline model was Logistic Regression.

Logistic Regression is a common baseline for text classification because it works efficiently with high-dimensional sparse TF-IDF representations.

The model was trained using the TF-IDF training matrix and evaluated on the validation set.

The validation performance was measured using:

* Accuracy
* Precision
* Recall
* Macro F1
* Weighted F1

The classification report was also generated to inspect performance across the eight product categories.

---

## 5. XGBoost

The second baseline model was XGBoost.

XGBoost was trained using the same TF-IDF feature representation and the same training/validation split used for Logistic Regression.

The configuration used for the XGBoost experiment was:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.1
random_state = 42
n_jobs = -1
```

Using the same TF-IDF representation allowed the two models to be compared under the same feature and dataset conditions.

---

## 6. Model Evaluation

Both models were evaluated on the same validation set.

The main metrics tracked were:

* **Accuracy**
* **Macro F1**
* **Weighted F1**
* Precision
* Recall

Macro F1 was included because it evaluates performance across all classes rather than allowing the overall metric to be dominated by individual class frequencies.

The validation results were used only for **model comparison and baseline selection**.

The untouched test set was not used to select the model.

---

## 7. MLflow Experiment Tracking

MLflow was introduced on Day 2 to track the baseline experiments.

Separate MLflow runs were created for Logistic Regression and XGBoost.

Each run recorded relevant information such as:

### Parameters

* Model type
* TF-IDF configuration
* Model hyperparameters

### Metrics

* Validation accuracy
* Validation precision
* Validation recall
* Validation macro F1
* Validation weighted F1

### Artifacts

Relevant experiment artifacts, such as classification reports and model-related files, were logged to MLflow.

This made it possible to compare the two experiments through the MLflow interface instead of relying only on notebook outputs.

---

## 8. Baseline Model Selection

After evaluating both classical models, XGBoost was selected as the classical baseline to carry forward.

The purpose of this selection was not to declare XGBoost the final production model at this stage.

Instead, XGBoost became the **baseline benchmark** for the next stage of the project.

The upcoming DistilBERT experiment would be trained and evaluated against this baseline using the same classification task and comparable evaluation metrics.

The baseline therefore established the question for the next stage:

> Can a fine-tuned transformer model improve upon the classical TF-IDF + XGBoost approach?

---

## 9. Model Artifact

The finalized XGBoost pipeline was saved so that the exact preprocessing and model combination could be reused later.

The pipeline contains the TF-IDF preprocessing step followed by the XGBoost classifier.

The saved artifact was:

```text
models/xgb_pipeline.pkl
```

The label encoder was saved separately:

```text
models/label_encoder.pkl
```

These artifacts represent the classical baseline that would be used for later comparison and, eventually, inference.

---

## 10. Day 2 Outcome

By the end of Day 2:

* The cleaned complaint text was converted into TF-IDF features.
* A 20,000-feature unigram + bigram representation was established.
* Logistic Regression was trained as the first classical baseline.
* XGBoost was trained as the second classical baseline.
* Both models were evaluated on the same validation set.
* MLflow was introduced for experiment tracking.
* XGBoost was selected as the classical baseline.
* The XGBoost pipeline and label encoder were saved as reusable artifacts.

The project now had a reliable classical NLP benchmark against which the upcoming **fine-tuned DistilBERT model** could be compared.

---

## Day 2 → Day 3

Day 2 established the classical baseline.

Day 3 moves to transformer-based NLP, where the goal is to fine-tune **DistilBERT** on the same complaint classification problem and compare its performance against the established TF-IDF + XGBoost baseline.

The comparison will use the same underlying classification task and held-out evaluation methodology to determine how the transformer approach performs relative to the classical NLP approach.

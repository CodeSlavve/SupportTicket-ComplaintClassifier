# Day 3 — DistilBERT Fine-Tuning

## Objective

Day 3 focused on moving from classical NLP to transformer-based text classification.

The objective was to fine-tune **DistilBERT** for CFPB complaint classification and evaluate whether a pretrained transformer could improve upon the TF-IDF + XGBoost classical baseline established on Day 2.

The transformer experiment was performed using Google Colab with GPU acceleration.

The same classification task and train/validation split were maintained so that the results could be compared fairly with the classical approach.

---

## 1. Dataset

The finalized dataset contained:

* **50,000 complaints**
* **8 product categories**
* **6,250 samples per category**
* `clean_complaint_text` — input text
* `Product` — target

The dataset was split into:

| Split      |    Samples |
| ---------- | ---------: |
| Training   |     35,000 |
| Validation |      7,500 |
| Test       |      7,500 |
| **Total**  | **50,000** |

The same training and validation split used for the classical baseline was used for DistilBERT.

The test set remained untouched and was reserved for final evaluation.

---

## 2. Transformer and Tokenizer

The pretrained:

```text
distilbert-base-uncased
```

model was selected for fine-tuning.

The corresponding DistilBERT tokenizer was used to convert complaint text into token IDs and attention masks.

The classification model was initialized using:

```text
AutoModelForSequenceClassification
```

with **8 output labels**, corresponding to the eight CFPB product categories.

---

## 3. Label Alignment

The same label mapping established during the classical modeling stage was preserved for the transformer experiment.

The `LabelEncoder` used during Day 2 was reused so that the numeric class IDs remained consistent between the classical and transformer models.

This allowed the predictions from both approaches to be interpreted using the same product-category names.

The encoder was retained as:

```text
models/label_encoder.pkl
```

---

## 4. Tokenization

The cleaned complaint narratives were tokenized using the DistilBERT tokenizer.

Each complaint was converted into the inputs required by the transformer:

* Input IDs
* Attention masks

Longer complaints were truncated according to the configured maximum sequence length, while shorter complaints were padded as required.

This transformed the raw complaint text into a format that could be processed by DistilBERT.

---

## 5. Fine-Tuning

The pretrained DistilBERT model was fine-tuned on the CFPB complaint training dataset.

The training process used:

* DistilBERT pretrained weights
* 35,000 training complaints
* 7,500 validation complaints
* 8 classification labels
* Google Colab GPU acceleration

The goal was to adapt DistilBERT's pretrained language representations to the specific financial-product classification task.

Unlike the classical TF-IDF approach, DistilBERT can use contextual information from surrounding words when determining the meaning of a complaint.

---

## 6. Validation Results

After fine-tuning, DistilBERT was evaluated on the **7,500-sample validation set**.

### Overall results

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **85.12%** |
| Macro F1    | **85.09%** |
| Weighted F1 | **85.09%** |

The transformer therefore established an **85.12%** validation benchmark, with **85.09%** Macro F1.

---

## 7. Per-Class Results

The validation classification report showed the following performance:

| Product Category                                        | Precision | Recall |       F1 | Support |
| ------------------------------------------------------- | --------: | -----: | -------: | ------: |
| Checking or savings account                             |      0.77 |   0.73 | **0.75** |     938 |
| Credit card                                             |      0.84 |   0.83 | **0.84** |     938 |
| Credit reporting or other personal consumer reports     |      0.88 |   0.92 | **0.90** |     937 |
| Debt collection                                         |      0.87 |   0.87 | **0.87** |     937 |
| Money transfer, virtual currency, or money service      |      0.79 |   0.82 | **0.81** |     937 |
| Mortgage                                                |      0.95 |   0.95 | **0.95** |     938 |
| Payday loan, title loan, personal loan, or advance loan |      0.81 |   0.80 | **0.81** |     937 |
| Vehicle loan or lease                                   |      0.90 |   0.88 | **0.89** |     938 |

The model performed particularly strongly on:

* Mortgage — **0.95 F1**
* Credit reporting — **0.90 F1**
* Vehicle loan or lease — **0.89 F1**
* Debt collection — **0.87 F1**

The lower-performing categories included checking/savings accounts, money transfers, and payday/title/personal/advance loans.

---

## 8. Comparison with the Classical Baseline

The purpose of Day 3 was to establish a transformer benchmark against the classical model selected on Day 2.

The comparison was based on the same validation set.

| Model            |   Accuracy |   Macro F1 | Weighted F1 |
| ---------------- | ---------: | ---------: | ----------: |
| TF-IDF + XGBoost |    ~83.32% |      83.35 |       83.35 |
| **DistilBERT**   | **85.12%** | **85.09%** |  **85.09%** |

The XGBoost validation accuracy was approximately **83.32%**, while DistilBERT achieved **85.12%** validation accuracy.

This gave DistilBERT a validation accuracy improvement of approximately **1.8 percentage points**.

DistilBERT achieved the higher validation Macro F1, establishing the transformer benchmark for the subsequent model-selection analysis.

The final model decision was still reserved for the later evaluation stage rather than being based solely on the validation set.

---

## 9. Saving the Fine-Tuned Model

After training, the fine-tuned DistilBERT model and tokenizer were saved using Hugging Face's `save_pretrained()` functionality.

The model was stored under:

```text
models/distilbert/
```

> *This folder, just like any other model.pkl file is not being tracked by github, It will only appear in your local machine if you train it.*

The saved directory contains the files required to reload the fine-tuned transformer, including:

```text
config.json
model.safetensors
tokenizer files
```

The label encoder remained separate:

```text
models/label_encoder.pkl
```

The DistilBERT model was retained locally for evaluation and model comparison. It was not included as a Git-tracked production artifact.

---

## 10. MLflow

The key results from the DistilBERT experiment were recorded in the project's MLflow experiment for comparison with the classical model runs.

The important information from the transformer experiment included:

* Model name
* Training configuration
* Number of classes
* Validation accuracy
* Validation Macro F1
* Validation weighted F1
* Classification report
* Confusion matrix
* Saved model artifacts

The experiment could then be compared against the classical MLflow runs from Day 2.

---
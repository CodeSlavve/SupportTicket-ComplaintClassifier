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
* `cleaned_complaint_text` — input text
* `Products` — target

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
| Accuracy    | **85.00%** |
| Macro F1    | **85.00%** |
| Weighted F1 | **85.00%** |

The transformer therefore established an **85% validation benchmark** for the project.

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
| TF-IDF + XGBoost |    ~83.32% |      83.34 |       83.34 |
| **DistilBERT**   | **85.00%** | **85.00%** |  **85.00%** |

The XGBoost validation accuracy was approximately **83.32%**, while DistilBERT achieved **85.00%** validation accuracy.

This gave DistilBERT a validation accuracy improvement of approximately **1.68 percentage points**.

More importantly, the transformer established a higher validation Macro F1 benchmark than the classical approach, making it the stronger candidate based on the validation results.

The final model decision was still reserved for the later evaluation stage rather than being based solely on the validation set.

---

## 9. Saving the Fine-Tuned Model

After training, the fine-tuned DistilBERT model and tokenizer were saved using Hugging Face's `save_pretrained()` functionality.

The model was stored under:

```text
models/distilbert/
```

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

This means the complete transformer inference setup consists of the DistilBERT model/tokenizer directory together with the label encoder.

---

## 10. MLflow

The DistilBERT experiment was prepared for integration with the project's MLflow experiment tracking.

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

## 11. Day 3 Outcome

By the end of Day 3:

* DistilBERT was fine-tuned for the 8-class CFPB complaint classification problem.
* The same 35,000/7,500 train-validation split was used.
* The same label mapping was maintained.
* The model achieved **85.00% validation accuracy**.
* The model achieved **85.00% Macro F1**.
* The model achieved **85.00% Weighted F1**.
* Per-class performance was evaluated.
* The fine-tuned model and tokenizer were saved.
* The transformer established a stronger validation benchmark than the classical XGBoost approach.

The project now had two trained approaches:

```text
Classical NLP
TF-IDF → XGBoost
~83.32% validation accuracy

Transformer NLP
DistilBERT
85.00% validation accuracy
```

---

## Day 3 → Next Stage

With both approaches trained and evaluated on the validation set, the next stage was to perform the final model evaluation and deeper analysis.

The untouched **7,500-sample test set** would be used only after the model was finalized.

This would provide the final generalization measurement and prevent the test set from influencing model selection.

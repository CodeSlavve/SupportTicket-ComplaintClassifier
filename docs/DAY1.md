# Day 1 — Problem Framing + EDA

## Objective

The first day focused entirely on understanding and preparing the CFPB consumer complaint dataset before training any machine learning model.

The goal was to convert the large raw CFPB complaint archives into a clean, manageable, balanced dataset containing only complaints with usable narrative text and a finalized set of product categories.

No machine learning models were trained on Day 1.

---

## 1. Problem Definition

The project aims to automatically classify a consumer complaint into its corresponding CFPB `Product` category based on the complaint narrative.

### Input

A consumer complaint narrative in natural language.

### Output

One of the finalized CFPB product categories.

This makes the task a **multi-class text classification problem**.

---

## 2. CFPB Dataset

The source data was obtained from the Consumer Financial Protection Bureau (CFPB) consumer complaint database.

The original CFPB data is very large, so using the complete dataset locally was unnecessary for this project. Instead, relevant rows were sampled from the CFPB archive while preserving the required product categories.

The final working dataset contains:

* **50,000 complaints**
* **16 original columns**
* **8 product categories**
* **50,000 non-null complaint narratives**

The main text field used for classification is:

```text
Consumer complaint narrative
```

The target field is:

```text
Product
```

---

## 3. Selecting the Product Categories

The initial CFPB data contains many different product categories. For this project, the classification problem was restricted to the following eight categories:

1. Mortgage
2. Debt collection
3. Credit reporting or other personal consumer reports
4. Vehicle loan or lease
5. Payday loan, title loan, personal loan, or advance loan
6. Credit card
7. Money transfer, virtual currency, or money service
8. Checking or savings account

These categories were selected from the available CFPB product data and used consistently throughout the project.

---

## 4. Handling Missing Complaint Narratives

A complaint cannot be classified from its text if the `Consumer complaint narrative` field is missing.

Therefore, rows with null complaint narratives were excluded from the modeling dataset.

The final dataset contains:

```text
Consumer complaint narrative: 50,000 non-null
```

This ensured that every row used for text classification had an actual complaint narrative.

---

## 5. Creating a Balanced Dataset

The raw CFPB data was not evenly distributed across the selected categories.

Some categories had substantially more available complaints than others. To prevent the final dataset from being dominated by the largest categories, an equal number of complaints was selected from each of the eight categories.

### Final class distribution

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

The resulting dataset is therefore perfectly balanced across the eight target classes.

---

## 6. Final Dataset Structure

After filtering and category selection, the sampled CFPB dataset contained 50,000 complaints across the eight selected product categories.

The working source dataset retained the relevant CFPB fields required during the data preparation stage.

During preprocessing, the data was reduced to the two fields required for model training:

### Modeling Dataset Schema

| Column | Purpose |
| :--- | :--- |
| `clean_complaint_text` | Cleaned consumer complaint narrative used as the model input |
| `Product` | Target product category used for classification (8 classes) |

The modeling dataset therefore contains only the information required for the downstream text classification pipeline.

The sampled source dataset is stored at:

```text
data/cfpb_sample_50k.csv
```

---


## 7. Text Preprocessing

The original `Consumer complaint narrative` field was processed to create the final:

`clean_complaint_text`

The purpose of this preprocessing was to make the complaint text suitable for downstream NLP processing while retaining the information needed to identify the financial product involved.

The original raw narrative field was not retained in the final modeling dataset.

The final dataset therefore separates the two essential components of the problem:

clean_complaint_text  →  Input feature
Product                →  Target label

## 8. Final Data Quality Checks

Before moving to modeling, the final dataset was checked to ensure that:

- Complaint text was available for every sample.
- The target category was available for every sample.
- Only the eight finalized product categories were present.
- Each category contained exactly 6,250 samples.
- The final dataset contained exactly 50,000 rows.
- Unnecessary CFPB columns had been removed.

This produced a compact dataset specifically designed for the classification pipeline.

---

## 9. Key Findings from Day 1

The main findings from the initial dataset exploration were:

* The raw CFPB dataset is too large to process unnecessarily in its entirety for this project.
* Consumer complaint narratives are the primary source of information for classification.
* The original product distribution is highly uneven.
* Eight product categories were finalized for the classification task.
* A balanced dataset of 50,000 complaints was created.
* Every final sample contains a non-null complaint narrative.
* Complaint narratives vary significantly in length.
* Financial-product categories can have overlapping language, which is likely to create classification ambiguity.

---

### Final dataset

```text
Dataset:       CFPB Consumer Complaints
Samples:       50,000
Features:      2 columns
Target:        Product
Classes:       8
Narratives:    50,000 non-null
Distribution:  6,250 samples per class
Saved to:      data/cfpb_sample_50k.csv
```

The dataset was now ready for the next stage: converting complaint text into numerical features using **TF-IDF** and establishing machine-learning baselines.

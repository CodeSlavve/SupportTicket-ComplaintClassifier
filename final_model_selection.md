# Final Model Selection

The final production model was selected by considering both **predictive performance and deployment cost**, rather than choosing the model with the highest validation score alone.

Three models were evaluated:

| Model               | Validation Macro-F1 |  Model Size | Measured RAM | Inference Benchmark |
| ------------------- | ------------------: | ----------: | -----------: | ------------------: |
| Logistic Regression |              82.94% |     1.99 MB |     22.71 MB |             0.19 ms |
| **XGBoost**         |          **83.35%** | **6.33 MB** | **88.77 MB** |         **0.55 ms** |
| DistilBERT          |          **85.09%** |   256.12 MB |    703.36 MB |             0.82 ms |

DistilBERT achieved the highest validation Macro-F1 at **85.09%**, outperforming XGBoost by **1.75 percentage points**. However, this improvement came with substantially higher resource requirements.

Compared with XGBoost:

* DistilBERT's model size was approximately **40.5× larger**.
* Its measured RAM usage was approximately **7.9× higher**.
* The CPU validation run took approximately **26.5 minutes for 7,500 complaints**.
* XGBoost maintained a validation Macro-F1 of **83.35%** while remaining considerably smaller and simpler to deploy.

XGBoost also improved upon the Logistic Regression baseline:

* Logistic Regression Macro-F1: **82.94%**
* XGBoost Macro-F1: **83.35%**
* Improvement: **0.41 percentage points**

Therefore, XGBoost provided a practical middle ground between the lightweight Logistic Regression baseline and the higher-performing but substantially more resource-intensive DistilBERT model.

The production-model decision can therefore be summarized as:

> **XGBoost was selected as the production model because it provided a strong validation Macro-F1 of 83.35% while maintaining substantially lower storage and memory requirements than DistilBERT. Although DistilBERT achieved a higher Macro-F1 of 85.09%, its significantly higher resource requirements and slower CPU-based processing made XGBoost a more practical choice for the intended deployment.**

This is an **engineering tradeoff rather than a claim that XGBoost was the highest-performing model**. DistilBERT remains the highest-scoring validation model in the experiment, while XGBoost was selected as the final deployment model based on the balance between predictive performance and operational requirements.

Following this decision, XGBoost was locked as the production candidate. The previously untouched test set was then used for its final evaluation.

The final XGBoost test results were:

| Metric             | Test Result |
| ------------------ | ----------: |
| Accuracy           |  **83.23%** |
| Weighted Precision |  **83.37%** |
| Recall             |  **83.23%** |
| Macro F1           |  **83.26%** |
| Weighted F1        |  **83.26%** |

The final deployment artifacts are:

```text
models/
├── xgb_pipeline.pkl
└── label_encoder.pkl
```

`xgb_pipeline.pkl` contains the TF-IDF transformation and XGBoost classifier, while `label_encoder.pkl` converts the numeric model output back into the original product-category labels.

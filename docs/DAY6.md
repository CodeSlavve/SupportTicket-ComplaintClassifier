# Day 6 — Data & Model Versioning with DVC

## Objective

Version the dataset and final model artifacts separately from Git using DVC, while keeping Git lightweight and making the project reproducible across machines.

---

## What I Implemented

### 1. DVC Setup

Installed and initialized DVC in the repository.

* DVC version: `3.67.1`
* Repository initialized with `.dvc/`
* DVC configuration committed to Git

---

### 2. Dataset Versioning

The final CFPB complaint dataset was added to DVC:

```text
data/cfpb_sample_50k.csv
```

The dataset contains:

* 50,000 complaint records
* 16 columns
* Consumer complaint narratives are present for all sampled records
* 8 balanced product categories

The actual CSV is excluded from Git using `.gitignore`.

DVC maintains the corresponding pointer file:

```text
data/cfpb_sample_50k.csv.dvc
```

The pointer file is version-controlled through Git, while the actual dataset is stored in the DVC remote.

---

### 3. Model Artifact Versioning

The final production artifacts were added to DVC:

```text
models/xgb_pipeline.pkl
models/label_encoder.pkl
```

The corresponding DVC pointer files are:

```text
models/xgb_pipeline.pkl.dvc
models/label_encoder.pkl.dvc
```

The binary model files remain outside Git and are stored through DVC.

The XGBoost pipeline represents the final selected text classification model, while the label encoder is required to correctly map the model's encoded predictions back to the original product labels.

---

## 4. DVC Remote

A Google Drive remote was configured as the DVC storage backend.

Remote:

```text
Google Drive
```

The remote is configured as the default DVC remote.

This keeps large datasets and model binaries separate from the Git repository while still making them accessible for future machines or collaborators.

---

## 5. DVC Push

The tracked artifacts were successfully uploaded using:

```bash
dvc push
```

Result:

```text
Authentication successful.
Pushing
3 files pushed
```

The three tracked artifacts were:

```text
data/cfpb_sample_50k.csv
models/xgb_pipeline.pkl
models/label_encoder.pkl
```

---

## 6. Git vs DVC

The project now separates source code from large data/model artifacts.

### Git stores

```text
Source code
Configuration
Documentation
DVC pointer files
DVC configuration
Project metadata
```

### DVC stores

```text
cfpb_sample_50k.csv
xgb_pipeline.pkl
label_encoder.pkl
```

This avoids committing large binary files directly to Git while preserving the exact versions required to reproduce the project.

---

## 7. Reproducibility Workflow

A new machine can reconstruct the required dataset and model artifacts using Git and DVC.

The basic workflow is:

```bash
git clone https://github.com/CodeSlavve/SupportTicket-ComplaintClassifier.git
cd SupportTicket-ComplaintClassifier
dvc pull
```

`git clone` retrieves the source code and DVC pointer files.

`dvc pull` retrieves the corresponding data and model artifacts from the configured DVC remote.

---

## 8. Fresh Checkout Verification

The DVC remote was successfully tested by pushing the tracked artifacts to Google Drive.

The final verification is performed by removing the local copies of:

```text
data/cfpb_sample_50k.csv
models/xgb_pipeline.pkl
models/label_encoder.pkl
```

while keeping their `.dvc` pointer files.

Running:

```bash
dvc pull
```

should restore all three artifacts from the DVC remote.

### Verification

```text
dvc status
```

Expected result:

```text
Data and pipelines are up to date.
```

After the round-trip test, the dataset should retain its original shape:

```text
(50000, 16)
```

and both model artifacts should be restored successfully.

---

## 9. Result

Day 6 establishes reproducible data and model versioning for the project.

The repository no longer depends on large datasets and model binaries being manually stored on a developer's machine.

The reproducibility chain is now:

```text
Git
 ├── Source code
 ├── Project configuration
 ├── DVC pointer files
 └── Documentation
          │
          ▼
       DVC
          │
          ▼
    Google Drive
          │
          ├── CFPB dataset
          ├── XGBoost pipeline
          └── Label encoder
```

This allows the exact dataset and final model artifacts used by the project to be recovered using Git + DVC.

---

## Commands Used

```bash
pip install dvc
dvc init

dvc add data/cfpb_sample_50k.csv

dvc add models/xgb_pipeline.pkl models/label_encoder.pkl

dvc remote add -d gdrive gdrive://<folder-id>

dvc push
dvc status

dvc pull
```

---
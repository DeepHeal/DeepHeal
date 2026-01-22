# DeepHeal: Self-supervised representations of drug-response proteomics

**DeepHeal** learns low-dimensional, denoised representations of **drug-induced proteomic changes** and exports them as CSV files for downstream machine learning models (e.g., **GLMVQ**, **LightGBM**).

Unlike typical single-cell workflows focusing on batch correction, DeepHeal is applied here to:

- **Input**: Bulk or aggregated proteomics **log2 fold-change (log2FC)** data (drug-treated vs. control).
- **Task**: **Unsupervised dimensionality reduction** (without batch effect modeling).
- **Output**: Low-dimensional embeddings (CSV) for each drug condition.
- **Labels (external)**: Drug efficacy classes (`Drug_Class`) from a meta file, used **only** for downstream classification, not for DeepHeal training.

Internally, DeepHeal implements a minimalist neural network with a single hidden layer and **contrastive learning** to obtain biologically meaningful latent features from proteomic responses.

---

## 1. Tutorial

You can try DeepHeal directly in your browser using Google Colab. The tutorial covers synthetic data generation, model training, embedding extraction, and visualization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepHeal/DeepHeal/blob/main/tutorial.ipynb)

**Tutorial Content:**
- Generating synthetic proteomics data.
- Initializing and training the DeepHeal model.
- Extracting latent embeddings.
- Visualizing embeddings with PCA.

---

## 2. Installation

### 2.1 Clone the repository

```bash
git clone https://github.com/DeepHeal/DeepHeal.git
cd DeepHeal
```

### 2.2 Install dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

*DeepHeal is implemented in Python and requires PyTorch.*

---

## 3. Usage

### 3.1 Command Line Interface (CLI)

You can use the provided `trainer.py` script to train the model and generate embeddings from CSV files.

**Basic Usage:**

```bash
python trainer.py \
  --input data/proteomics_log2fc.csv \
  --id-col Sample_ID \
  --latent-dim 32 \
  --output embeddings/deepheal_latent_32d.csv \
  --no-batch
```

**Arguments:**

- `--input`: Path to the input CSV file containing log2 fold-changes.
- `--id-col`: Name of the column containing sample identifiers (e.g., `Sample_ID`).
- `--meta`: (Optional) Path to a metadata CSV file.
- `--latent-dim`: Size of the latent dimension (default: 32).
- `--output`: Path where the output embeddings CSV will be saved.
- `--no-batch`: Disables batch correction logic (recommended for simple dimensionality reduction).
- `--epochs`: Number of training epochs (default: 15).
- `--batch-size`: Batch size (default: 32).
- `--lr`: Learning rate (default: 1e-3).

### 3.2 Python API

You can also use DeepHeal programmatically within your Python scripts.

```python
import pandas as pd
from deepheal.deepheal import DeepHeal

# 1. Load Data
# Ensure your data is a DataFrame where rows are samples and columns are features
# 'Sample_ID' column should be handled separately or as index
df = pd.read_csv("data/proteomics_log2fc.csv")
features = df.drop(columns=["Sample_ID"])  # Drop non-feature columns

# 2. Initialize Model
model = DeepHeal(
    save_dir="output",
    latent_dim=32,
    n_epochs=20,
    batch_size=32,
    domain_key=None  # Set to None for no batch correction
)

# 3. Set Data & Train
model.set_data(features)
model.train(save_model=True)

# 4. Generate Embeddings
embeddings = model.predict(features)
print(f"Embeddings shape: {embeddings.shape}")
```

---

## 4. Data Format

### 4.1 Proteomics log2FC matrix

- **File**: e.g., `proteomics_log2fc.csv`
- **Format**: CSV
- **Structure**:
    - Each **row** = one sample / drug condition
    - Each **column** = one protein
    - Each cell = **log2FC** (treated vs. control)

**Example:**

```csv
Sample_ID,PROT1,PROT2,PROT3,...
DrugA,0.52,-0.13,1.07,...
DrugB,-0.95,0.20,-0.33,...
...
```

**Requirements:**

- Values must be **log2-transformed fold-changes**.
- `Sample_ID` must uniquely identify each sample (e.g., drug name or treatment ID).

### 4.2 Meta file: drug classes (Optional)

- **File**: e.g., `meta.csv`
- **Format**: CSV
- **Required columns**:
  - `Sample_ID`: Aligns with `Sample_ID` in the log2FC matrix.
  - `Drug_Class`: Pharmacological / efficacy class label.

**Example:**

```csv
Sample_ID,Drug_Class
DrugA,Kinase_inhibitor
DrugB,GPCR_antagonist
...
```

---

## 5. Downstream Classification (GLMVQ / LightGBM)

DeepHeal itself does **not** train supervised classifiers. You must export the embeddings and build your own models.

Below is a minimal Python example using **LightGBM**:

```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load embeddings and meta data
Z = pd.read_csv("embeddings/deepheal_latent_32d.csv")
meta = pd.read_csv("data/meta.csv")

# 2. Merge on Sample_ID
df = Z.merge(meta, on="Sample_ID")

# 3. Define Features (z1...zN) and Labels (Drug_Class)
feature_cols = [c for c in df.columns if c.startswith("z")]
X = df[feature_cols]
y = df["Drug_Class"]

# Note: You may need to encode 'y' to integers if not already done.
# y = y.astype('category').cat.codes

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "multiclass",
    "num_class": y.nunique(),  # Ensure this matches your class count
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=200,
    callbacks=[lgb.early_stopping(stopping_rounds=20)]
)

# 6. Evaluate
y_pred = model.predict(X_test)
y_pred_labels = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred_labels))
```

---

## 6. Method Summary

**DeepHeal Framework:**

- Uses a **single-hidden-layer** neural network.
- Trained via **self-supervised contrastive learning**.
- Learns latent representations that:
  - Denoise proteomic response profiles.
  - Preserve local and global data structure.
  - Capture biologically meaningful patterns (e.g., co-regulated proteins/pathways).

**In this repository:**

- We **do not** model or correct batch effects.
- We focus on using DeepHeal as a **general-purpose dimensionality reduction** tool for drug-response proteomics, preparing robust features for downstream machine learning.

---

## 7. License

MIT License

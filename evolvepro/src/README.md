# EvolvePro — Code Documentation

This package implements a machine-learning-guided directed evolution workflow for proteins. Given wet-lab activity measurements and protein language model (PLM) embeddings, it trains a regression model to predict which untested variants are most worth testing next.

---

## Package layout

```
evolvepro/src/
├── model.py      — first-round selection and regression model
├── evolve.py     — top-level orchestration
└── process.py    — FASTA generation utilities
```

---
## Module reference

### `evolve.py`

**`evolve(embeddings_files, round_files, output_file, regression_model='randomforest')`**

Top-level function. Loads activity and embedding data, trains a regression model on tested variants, then predicts activity for every variant in the embeddings matrix.

| Parameter  | Description                                  |
|---|----------------------------------------------|
| `embeddings_files` | Paths to embeddings CSV files                |
| `round_files` | Paths to activity Excel files                |
| `output_file` | Path to write the predictions CSV            |
| `regression_model` | Model to use (see options in `model.py` below) |


Returns a `DataFrame` with `rank`, `<label>_actual`, and `<label>_pred` columns, sorted by predicted activity (descending). Variants with no measurements have `NaN` in `_actual` columns.

The order of the entries in the embedding files and activity files does not matter. Passing entries in a single file or multiple files makes no difference.

---

### `model.py`

**`first_round(labels, embeddings, explicit_variants=None, num_mutants_per_round=16, first_round_strategy='random', embedding_type=None, random_seed=None)`**

Selects an initial panel of variants to test before any measurements exist. Three strategies are supported:

- `"random"` — uniform random sampling from all non-WT variants.
- `"diverse_medoids"` — K-medoids clustering in embedding space; returns one medoid per cluster to maximise sequence diversity. Applies PCA (10 components) before clustering unless `embedding_type="embeddings_pca"`. Requires `scikit-learn-extra`.
- `"explicit_variants"` — use a caller-supplied list of variant names.

- 
- #TODO: clean up first_round()

**`regression(embeddings, experimental_data, regression_model='randomforest')`**

Trains a regression model on tested variants and predicts activity for all variants in the embeddings matrix. Called internally by `evolve()` but can be used directly.

Supported `regression_model` values:

| Value | Model |
|---|---|
| `"ridge"` | Ridge regression (cross-validated alpha) |
| `"lasso"` | Lasso / MultiTask Lasso |
| `"elasticnet"` | Elastic net / MultiTask Elastic net |
| `"linear"` | Ordinary least squares |
| `"neuralnet"` | MLP (two hidden layers: 64→32 units, ReLU, Adam, early stopping) |
| `"randomforest"` | Random forest (200 trees) — **default** |
| `"gradientboosting"` | XGBoost |
| `"knn"` | k-nearest neighbours (k=10, distance-weighted, PCA pre-processing) |
| `"gp"` | Gaussian process (RBF + white noise kernel) |

---

### `process.py`

**`generate_wt(wt_sequence, output_file)`**

Writes a single-record FASTA with record ID `WT` from a protein sequence string.

**`generate_single_aa_mutants(wt_sequence, output_file)`**

Generates all possible single amino-acid substitutions across the full sequence and writes them to a FASTA file. Silent substitutions are skipped. Variant names follow standard notation: `<WT AA><position><mutant AA>` (1-based), e.g. `A107M`.

**`generate_multi_mutants(wt_sequence, output_file, min_mutations, max_mutations, mutations_list)`**

Given a list of mutations (e.g. from single-mutant screening), generates all valid combinatorial variants within the specified mutation count range. Combinations that mutate the same position twice are skipped. Multi-mutant names are underscore-joined: `A107M_F73C`.

| Parameter | Type | Description |
|---|---|---|
| `min_mutations` | `int` | Minimum number of mutations per variant |
| `max_mutations` | `int` | Maximum number of mutations per variant |
| `mutations_list` | `list[str]` | Mutations to combine, e.g. `['A107M', 'F73C']` |

---


## File format specifications

### Embeddings (.csv)

A numeric matrix where:
- The **leftmost column** contains variant names (row index). These must exactly match the names used in your activity files.
- All remaining columns are embedding dimensions (all numeric, no missing values).
- Every variant in your activity files must have a row here. Extra rows (untested variants) are fine — they form the prediction pool.

```
           dim_0   dim_1  ...  dim_1279
WT         0.123   0.456  ...  0.789
A107M      0.111   0.222  ...  0.333
F73C       0.234   0.567  ...  0.012
```

Embeddings are produced by running a PLM on your mutant FASTA file. Refer to `README_plm.md` for extraction scripts.

---

### Activity data (`.xlsx`)

Excel files where:
- The **leftmost column** contains variant names (used as the row index). These must exactly match the variant names used in the embeddings file.
- Every additional column is treated as an activity measurement dimension. Column names become the output labels (e.g. a column named `activity` produces `activity_pred` and `activity_actual` in the output).
- One file per experimental round; multiple round files are concatenated automatically.

```
variant    activity
WT         1.0
A107M      1.8
F73C       0.6
```

You can include multiple activity columns for multi-objective optimisation — the model will predict each independently and rank variants by their sum.


## Output

`evolve()` writes a CSV and returns the same data as a `DataFrame`, sorted by predicted activity (descending):

| Column | Description |
|---|---|
| `rank` | 1-based ranking by predicted activity |
| `<label>_actual` | Measured activity; `NaN` for untested variants |
| `<label>_pred` | Predicted activity for every variant in the embeddings |
| `sum_pred` | Sum across all predicted dimensions (multi-output only) |




## Typical workflow

### Step 1 — Generate variant FASTA files

```python
from evolvepro.src.process import generate_wt, generate_single_aa_mutants

generate_wt('MKTAYIAKQR...', 'WT.fasta')
generate_single_aa_mutants('MKTAYIAKQR...', 'mutants.fasta')
```

### Step 2 — Extract PLM embeddings

Pass `WT.fasta` & `mutants.fasta` to a PLM extraction script (see `README_plm.md`). The output CSV must have variant names as the row index.

### Step 3 — Select initial variants (no measurements yet)

```python
from evolvepro.src.model import first_round
import pandas as pd

labels = pd.DataFrame({'variant': all_variant_names})
embeddings = pd.read_csv('embeddings/mutants.csv', index_col=0)

labels_zero, iteration_zero, to_test = first_round(
    labels=labels,
    embeddings=embeddings,
    num_mutants_per_round=16,
    first_round_strategy='diverse_medoids',
    random_seed=42,
)
print(to_test)  # variants to send to the lab
```

### Step 4 — Predict after wet-lab measurements

Once activity measurements are in hand (saved as `round_1.xlsx`):

```python
from evolvepro.src.evolve import evolve

predictions = evolve(
    embeddings_files=['embeddings/mutants.csv'],
    round_files=['experimental_data/round_1.xlsx'],
    output_file='outputs/round_1_predictions.csv',
)
```

### Step 5 — Combine beneficial mutations (optional)

```python
from evolvepro.src.process import generate_multi_mutants

generate_multi_mutants(
    wt_sequence='MKTAYIAKQR...',
    output_file='multi_mutants.fasta',
    min_mutations=2,
    max_mutations=2,
    mutations_list=['A107M', 'F73C', 'K201R'],
)
```

Embed `multi_mutants.fasta` with the same PLM, then pass both embedding files and all round files to `evolve()`:

```python
predictions = evolve(
    embeddings_files=[
        'embeddings/mutants.csv',
        'embeddings/multi_mutants.csv',
    ],
    round_files=[
        'experimental_data/round_1.xlsx',
        'experimental_data/round_2.xlsx',
    ],
    output_file='outputs/multi_round_predictions.csv',
)
```

> **Note:** When passing multiple embedding files, use the same PLM and extraction settings for all — the files must have the same number of dimensions.

---


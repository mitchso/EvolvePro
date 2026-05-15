# EvolvePro — Code Documentation

This package implements a machine-learning-guided directed evolution workflow for proteins. Given wet-lab measurements of variant activity and protein language model embeddings, it trains a regression model to predict which untested variants to prioritise in the next experimental round.

---

## Package layout

```
evolvepro/src/
├── model.py      — first-round selection and active-learning regression model
├── evolve.py     — top-level orchestration for experimental workflows
└── process.py    — pre-processing and FASTA generation utilities
```

---

## Quick start

```python
from evolvepro.src.evolve import evolve

predictions = evolve(
    round_name='round_1',
    embeddings_files=['embeddings/mutants_esm2_t33_650M_UR50D.csv'],
    round_files=['experimental_data/round_1.xlsx'],
    output_dir='outputs/',
)
```

---

## File format specifications

### 1. Wild-type FASTA (`WT.fasta`)

A standard single-record FASTA file containing the full wild-type protein sequence.

```
>WT
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG...
```

Generate this automatically with `generate_wt()` from `process.py`. The record ID must be exactly `WT`.

---

### 2. Mutant FASTA (`mutants.fasta`)

A multi-record FASTA file listing every candidate variant plus WT. Each record ID must be a variant name in standard single-mutant notation: `<WT AA><position><mutant AA>` (1-based), e.g. `A107M`.

```
>WT
MKTAYIAKQR...
>A107M
MKTAYIAKQR...(sequence with A→M at position 107)...
>F73C
...
```

Multi-mutant variants use underscore-joined tokens: `A107M_F73C`.

Generate this file automatically with `generate_single_aa_mutants()` or `generate_n_mutant_combinations()` from `process.py`.

---

### 3. Embeddings CSV

A numeric matrix where rows are variants (including `WT`) and columns are embedding dimensions. The first (unnamed) column is the row index of variant names.

```
,dim_0,dim_1,...,dim_1279
WT,0.123,0.456,...,0.789
A107M,0.111,0.222,...,0.333
```

**Rules:**
- Row index values must exactly match the variant names used in round files.
- All values must be numeric; no missing entries are allowed.
- Every variant that appears in any round file must have a row in the embeddings CSV.
- Variants present in the embeddings but absent from round files are treated as the unlabelled prediction pool — this is expected and correct.

Embeddings are produced by running a protein language model on `mutants.fasta`. Refer to `README_plm.md` for extraction scripts.

---

### 4. Round files (experimental data)

Excel files (`.xlsx`) where:
- The first (unnamed) column is the row index of variant names.
- The first data column is the activity measurement (labelled `activity`).
- One file per experimental round.

```
variant,activity
WT,1.0
A107M,1.8
F73C,0.6
```

---

## Module reference

### `evolve.py`

**`evolve(round_name, embeddings_files, round_files, output_dir=None)`**

Top-level function that loads experimental and embedding data, trains a regression model, and writes predictions for all variants to a CSV.

| Parameter | Type | Description |
|---|---|---|
| `round_name` | `str` | Prefix for the output filename |
| `embeddings_files` | `list[str]` | Paths to embeddings CSV files (concatenated in order) |
| `round_files` | `list[str]` | Paths to experimental round Excel files (concatenated in order) |
| `output_dir` | `str \| None` | Directory to write `<round_name>_predictions.csv`; defaults to current directory |

Returns a `DataFrame` with columns `y_pred` (predicted activity for all variants) and `y_actual` (measured activity; `NaN` for untested variants).

Raises `ValueError` if duplicate variant names are found in either the experimental data or embeddings after concatenation.

**`validate_csv(file_path, file_type)`**

Loads and validates a CSV or Excel file.

- `file_type="embeddings"` — reads with `pd.read_csv`, variant name as index.
- `file_type="experimental"` — reads with `pd.read_excel`, variant name as index.

Returns a `DataFrame` sorted by index (variant name).

---

### `model.py`

**`first_round(labels, embeddings, explicit_variants=None, num_mutants_per_round=16, first_round_strategy='random', embedding_type=None, random_seed=None)`**

Selects the initial panel of variants to test before any measurements exist. Three strategies are available:

- `"random"` — uniform random sampling from all non-WT variants.
- `"diverse_medoids"` — K-medoids clustering in embedding space; returns the medoid of each cluster to maximise sequence space coverage. Applies PCA (10 components) before clustering unless `embedding_type="embeddings_pca"`. Requires the `scikit-learn-extra` package.
- `"explicit_variants"` — use a caller-supplied list of variant names.

Returns `(labels_zero, iteration_zero, this_round_variants)`:
- `labels_zero` — `labels` left-joined with iteration assignments.
- `iteration_zero` — DataFrame with `variant` and `iteration` columns; iteration is `0` for all selected variants and for WT.
- `this_round_variants` — Series of selected variant names.

Raises `ValueError` if `num_mutants_per_round` exceeds the number of available non-WT variants, or if `first_round_strategy="explicit_variants"` is chosen without supplying `explicit_variants`.

**`regression(embeddings, experimental_data, regression_type='randomforest', random_seed=None)`**

Trains a regression model on tested variants and predicts activity for all variants in the embeddings matrix. The first column of `experimental_data` is used as the target (`y`).

Supported `regression_type` values:

| Value | Model |
|---|---|
| `"ridge"` | Ridge regression (cross-validated alpha) |
| `"lasso"` | Lasso regression (cross-validated alpha) |
| `"elasticnet"` | Elastic net (cross-validated) |
| `"linear"` | Ordinary least squares |
| `"neuralnet"` | MLP regressor (1 hidden layer, 5 units, ReLU, Adam) |
| `"randomforest"` | Random forest (100 trees, Friedman MSE) — **default** |
| `"gradientboosting"` | XGBoost regressor |
| `"knn"` | k-nearest neighbours (k=5) |
| `"gp"` | Gaussian process regressor |

Returns a `DataFrame` indexed by variant name with columns `y_pred` and `y_actual`.

**`_build_model(regression_type)`**

Internal helper. Instantiates and returns an unfitted regression estimator for the given `regression_type`. Raises `ValueError` for unrecognised types.

---

### `process.py`

**`generate_wt(wt_sequence, output_file)`**

Writes a single-record FASTA file containing the wild-type sequence with record ID `WT`. This file is the starting point for all downstream FASTA generation.

| Parameter | Type | Description |
|---|---|---|
| `wt_sequence` | `str` | Full wild-type protein sequence (one-letter amino acid codes) |
| `output_file` | `str` | Destination path, e.g. `"WT.fasta"` |

---

**`generate_single_aa_mutants(wt_fasta, output_file, positions=None)`**

Generates all possible single amino-acid substitutions across the full sequence (or a restricted set of positions) and writes them to a FASTA file. The WT sequence is included as the first record.

Variant names follow the convention `<WT AA><position><mutant AA>` (1-based), e.g. `A107M`. Silent substitutions (mutant AA = WT AA) are skipped.

| Parameter | Type | Description |
|---|---|---|
| `wt_fasta` | `str` | Path to the wild-type FASTA file |
| `output_file` | `str` | Destination path for the mutant FASTA |
| `positions` | `list[int] \| None` | 1-based positions to mutate; `None` mutates all positions |

---

**`generate_n_mutant_combinations(wt_fasta, mutant_file, n, output_file, threshold=1.0)`**

Generates all valid n-fold combinations of beneficial single mutants and writes them to a FASTA file. Reads single-mutant activity from an Excel file, filters to variants with `activity > threshold`, and enumerates every combination of `n` mutations. Combinations that apply two mutations to the same position are skipped.

Multi-mutant variant names join single-mutant tokens with underscores: `A107M_F73C`.

| Parameter | Type | Description |
|---|---|---|
| `wt_fasta` | `str` | Path to the wild-type FASTA file |
| `mutant_file` | `str` | Path to an Excel file with `Variant` (position-only format, e.g. `107M`) and `activity` columns |
| `n` | `int` | Number of mutations to combine (e.g. `2` for double mutants) |
| `output_file` | `str` | Destination path for the output FASTA |
| `threshold` | `float` | Minimum activity score for inclusion (default `1.0`) |

---

## Typical workflow

### Step 1 — Generate variant FASTA files

```python
from evolvepro.src.process import generate_wt, generate_single_aa_mutants

generate_wt('MKTAYIAKQR...', 'WT.fasta')
generate_single_aa_mutants('WT.fasta', 'mutants.fasta')
```

### Step 2 — Extract protein language model embeddings

Pass `mutants.fasta` to a PLM extraction script (see `README_plm.md`). The output CSV must have variant names as the row index.

```bash
python extract_esm.py esm2_t33_650M_UR50D mutants.fasta esm_out/ \
    --include mean \
    --repr_layers -1 \
    --concatenate_dir embeddings/
# produces: embeddings/mutants_esm2_t33_650M_UR50D.csv
```

### Step 3 — Select initial variants (no measurements yet)

```python
from evolvepro.src.model import first_round
import pandas as pd

labels = pd.DataFrame({'variant': all_variant_names})
embeddings = pd.read_csv('embeddings/mutants_esm2_t33_650M_UR50D.csv', index_col=0)

labels_zero, iteration_zero, to_test = first_round(
    labels=labels,
    embeddings=embeddings,
    num_mutants_per_round=16,
    first_round_strategy='diverse_medoids',
    random_seed=42,
)
print(to_test)  # variants to send to the lab
```

### Step 4 — Run predictions after wet-lab measurements

Once you have activity measurements back from the lab (saved as `round_1.xlsx`):

```python
from evolvepro.src.evolve import evolve

predictions = evolve(
    round_name='round_1',
    embeddings_files=['embeddings/mutants_esm2_t33_650M_UR50D.csv'],
    round_files=['experimental_data/round_1.xlsx'],
    output_dir='outputs/',
)
# writes: outputs/round_1_predictions.csv
# returns: DataFrame with y_pred for all variants, y_actual for tested ones
```

### Step 5 — Combine mutations (optional)

After identifying beneficial single mutants, generate and evaluate multi-mutant combinations:

```python
from evolvepro.src.process import generate_n_mutant_combinations

generate_n_mutant_combinations(
    wt_fasta='WT.fasta',
    mutant_file='beneficial_singles.xlsx',
    n=2,
    output_file='multi_mutants.fasta',
    threshold=1.5,
)
```

Embed `multi_mutants.fasta` with the same PLM, then pass both embedding files to `evolve`:

```python
predictions = evolve(
    round_name='multi_round_1',
    embeddings_files=[
        'embeddings/mutants_esm2_t33_650M_UR50D.csv',
        'embeddings/multi_mutants_esm2_t33_650M_UR50D.csv',
    ],
    round_files=[
        'experimental_data/round_1.xlsx',
        'experimental_data/round_2.xlsx',
        'experimental_data/multi_round_1.xlsx',
    ],
    output_dir='outputs/',
)
```

> **Note:** When passing multiple embedding files, use the same PLM and extraction layer for both — the files must have the same number of dimensions.

---

## Output

`evolve()` writes `<output_dir>/<round_name>_predictions.csv` and returns the same data as a `DataFrame`:

| Column | Description |
|---|---|
| `y_pred` | Predicted activity for every variant in the embeddings |
| `y_actual` | Measured activity; `NaN` for variants not yet tested |
# EvolvePro — Code Documentation

This package implements a machine-learning-guided directed evolution workflow for proteins.
It supports two modes of operation:

- **Experimental mode** — you have real wet-lab measurements and want the model to predict
  which untested variants to try next.
- **DMS simulation mode** — you have a deep mutational scanning (DMS) dataset and want to
  benchmark how well different active-learning strategies would have navigated it.

---

## Package layout

```
evolvepro/src/
├── data.py       — file loading and alignment helpers
├── model.py      — first-round selection and active-learning model
├── evolve.py     — top-level orchestration (experimental workflows)
├── process.py    — pre-processing raw DMS / experimental data; FASTA generation
├── plot.py       — plotting utilities for experimental results
├── plot_dms.py   — plotting utilities for DMS simulation results
└── utils.py      — shared helpers (PCA wrapper)
```

---

## File format specifications

### 1. Wild-type FASTA (`WT.fasta`)

A standard single-record FASTA file containing the full wild-type protein sequence.

```
>WT Wild-type sequence
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
```

- The sequence ID can be anything; the code reads only the first record.
- Used to reconstruct mutant sequences from variant names in `process_dataset()` and
  `generate_n_mutant_combinations()`.
- `generate_wt()` writes this file automatically, producing a record with ID `WT`.

---

### 2. Mutant FASTA (`mutants.fasta`)

A multi-record FASTA file listing every candidate variant (and optionally WT).
Each record ID must be a **variant name** in standard single-mutant notation (see §5).

```
>WT
MKTAYIAKQR...
>A107M
MKTAYIAKQR...(sequence with A→M at position 107)...
>F73C
...
```

- Generated automatically by `generate_single_aa_mutants()` or `generate_n_mutant_combinations()`.
- Used by `suggest_initial_mutants()` to randomly sample a starting panel.

---

### 3. Embeddings CSV (`mutants_esm2_t33_650M_UR50D.csv`)

A numeric matrix where:
- **Rows** = variants (including `WT`)
- **Columns** = embedding dimensions (one per feature; headers can be anything)
- **Row index** (first column, unnamed) = variant name

```
,dim_0,dim_1,dim_2,...,dim_1279
WT,0.123,0.456,...,0.789
A107M,0.111,0.222,...,0.333
F73C,...
```

**General rules (apply to all workflows):**
- The row index values must exactly match the variant names used in the experimental
  round files.
- All values must be numeric; no missing entries are allowed.
- Embeddings are produced by running a protein language model on `mutants.fasta` using
  one of the `extract_*.py` scripts (see `README_plm.md`). Every script outputs a CSV
  in this exact format.

---

#### Preparing embeddings for `evolve_experimental()`

`evolve_experimental` takes a **single** embeddings CSV covering your entire
single-mutant candidate library plus WT. The typical preparation steps are:

**Step 1 — Generate a combined FASTA.**
`mutants.fasta` must contain one record for WT and one record for every single-mutant
candidate you want to evaluate (i.e. every variant the model could potentially recommend
for the next round, not just the ones already tested). Use `generate_wt()` and
`generate_single_aa_mutants()` from `process.py` to build this file automatically.

**Step 2 — Run a PLM extraction script.**
Pass `mutants.fasta` to any of the `extract_*.py` scripts. For example, using ESM-2:

```bash
python extract_esm.py esm2_t33_650M_UR50D mutants.fasta esm_out/ \
    --include mean \
    --repr_layers -1 \
    --concatenate_dir embeddings/
# produces: embeddings/mutants_esm2_t33_650M_UR50D.csv
```

All extraction scripts produce a CSV with the FASTA record ID as the row index.
Since `mutants.fasta` uses variant names as record IDs (e.g. `>A107M`), the row
index of the output CSV will already be in the correct format. Ensure the WT record
in your FASTA uses the ID `WT` exactly — `validate_csv` requires the index value
`"WT"` and will accept no other form.

**Step 3 — Verify coverage.**
Every variant name that appears in any of your round files must have a corresponding
row in the embeddings CSV, and conversely every candidate you might want recommended
must also have a row (the model can only score variants it has an embedding for).
Variants present in the embeddings but absent from round files are treated as the
unlabelled prediction pool — this is expected and correct.

**Quick checklist for `evolve_experimental`:**

| Check | What to verify |
|---|---|
| Single CSV | One file covering WT + all single-mutant candidates |
| Row index = variant name | e.g. `WT`, `A107M`, `F73C` — not FASTA descriptions |
| WT row present | Required; the index value must be exactly `"WT"` |
| No missing values | All embedding dimensions must be numeric |
| Coverage | Every tested and candidate variant has a row |

---

#### Preparing embeddings for `evolve_experimental_multi()`

`evolve_experimental_multi` accepts a **list** of embeddings CSV files
(`embeddings_files`). This is necessary because single-mutant and multi-mutant
libraries are typically embedded separately — the multi-mutant FASTA is only generated
after the single-mutant rounds have identified the best positions to combine.

The files are concatenated in the order they appear in the list. One important
constraint: **WT must appear in exactly one file** (the first). Any row named `"WT"`
in the second or later files triggers a warning and is dropped before concatenation to
prevent duplicate index entries.

**Typical setup — two embeddings files:**

```
embeddings_files=[
    'mutants_esm2_t33_650M_UR50D.csv',        # single-mutant library — MUST contain WT
    'multi_mutants_esm2_t33_650M_UR50D.csv',  # multi-mutant library — must NOT contain WT
]
```

**Step 1 — Embed the single-mutant library (same as `evolve_experimental`).**
Follow steps 1–2 above for `mutants.fasta`. This file should contain WT and all
single-mutant candidates. Use this as the **first** entry in `embeddings_files`.

**Step 2 — Generate and embed the multi-mutant library.**
Once you have identified promising single mutations, generate a combinatorial FASTA
(e.g. with `generate_n_mutant_combinations()` from `process.py`) containing all
multi-mutant candidates you want to evaluate. **Do not include WT** in this FASTA —
if it ends up in the CSV anyway, it will be dropped automatically with a warning,
but omitting it avoids unnecessary computation.

Run the same PLM extraction script on this FASTA:

```bash
python extract_esm.py esm2_t33_650M_UR50D multi_mutants.fasta esm_out_multi/ \
    --include mean \
    --repr_layers -1 \
    --concatenate_dir embeddings/
# produces: embeddings/multi_mutants_esm2_t33_650M_UR50D.csv
```

**Step 3 — Use the same PLM and layer for both files.**
The single-mutant and multi-mutant CSVs are concatenated into a single embedding
matrix before training. The two files must therefore have the **same number of
columns** — i.e. they must come from the same model and the same extraction layer.
Mixing, for example, ESM-2 650M embeddings (1280 dims) with ProtT5 embeddings
(1024 dims) will raise a `ValueError` on the inconsistent dimensionality.

**Step 4 — Ensure row index format matches round file variant names.**
Multi-mutant variant names must use underscore-joined standard notation, e.g.
`A107M_F73C`. The record IDs in your multi-mutant FASTA must already be in this
format, so that the row index of the CSV matches the `variant` column in the
multi-mutant round files.

**Quick checklist for `evolve_experimental_multi`:**

| Check | What to verify |
|---|---|
| Same PLM + layer for all files | Ensures consistent embedding dimensionality |
| WT in first file only | WT rows in subsequent files are dropped automatically |
| Multi-mutant row index = full variant name | e.g. `A107M_F73C`, not `107M_73C` |
| No missing values in any file | All embedding dimensions must be numeric |
| Coverage | Every candidate (single- and multi-mutant) has a row in one of the files |

---

### 4. Experimental round file (`D4_round_N.xlsx`)

An Excel workbook (`.xlsx`) with **at least** these two columns (additional columns are
preserved in processing):

| Column     | Type   | Description |
|------------|--------|-------------|
| `variant`  | string | Variant identifier in standard notation (see §5) |
| `activity` | float  | Measured activity value for that variant |

Column names are matched case-insensitively, so `Variant`, `VARIANT`, and `variant`
are all accepted.

**Rules:**
- One row per tested variant.
- WT must appear in round 1 as `variant = "WT"`. If absent, a warning is issued.
- WT should **not** be re-listed in subsequent rounds; any WT row in rounds 2+ is
  dropped with a warning.
- No duplicate variants are permitted across all rounds — the code raises a
  `ValueError` and prints all duplicates if any are found.

**Variant column format:**

The `variant` column must contain variant names in fully specified standard notation
(e.g. `"A107M"`, `"A107M_F73C"`) or `"WT"`. Partial / position-only formats
(e.g. `"107M"`) are not accepted and will raise a `ValueError` during validation.

---

### 5. Variant name convention

Throughout the codebase, variants are represented as:

```
<WT amino acid><1-based position><mutant amino acid>
```

Examples: `A107M`, `F73C`, `K138S`

Multi-mutants join single-mutant tokens with underscores: `A107M_F73C`

The wild-type is always the string `"WT"`.

---

### 6. DMS labels CSV (`<dataset_name>_labels.csv`)

Produced by `process_dataset()` and consumed by the simulation pipeline.

| Column           | Type   | Description |
|------------------|--------|-------------|
| `variant`        | string | Variant name in standard notation |
| `activity`       | float  | Raw activity value |
| `activity_scaled`| float  | Min-max scaled activity (0–1) |
| `activity_binary`| int    | 1 if activity exceeds the chosen cutoff, else 0 |

Additional `activity_binary_<p>p` columns may be present if percentile cutoffs were
requested via `cutoff_percentiles`.

---

## Typical experimental workflow

```
1. generate_wt()                    →  WT.fasta
2. generate_single_aa_mutants()     →  mutants.fasta
3. <Run ESM / language model>       →  mutants_esm2_t33_650M_UR50D.csv
4. suggest_initial_mutants()        →  pick first N variants to test
5. <Run wet-lab assay>              →  D4_round_1.xlsx
6. evolve_experimental()            →  ranked predictions for next round
7. <Run wet-lab assay>              →  D4_round_2.xlsx
8. evolve_experimental()            →  ... repeat
```

---

## Output files (written by `evolve_experimental`)

Each call to `evolve_experimental()` creates a subdirectory `<output_dir>/<round_name>/`
containing four CSV files. The primary file for planning the next round is
**`df_test.csv`** — sort by `y_pred` descending and pick from the top.

---

### `iteration.csv`

Records which assay round each variant was tested in. Contains only variants that have
been measured (i.e. the training set); untested variants are not listed.

| Column      | Type  | Description |
|-------------|-------|-------------|
| `variant`   | str   | Variant name in standard notation (e.g. `A107M`) |
| `iteration` | float | Round number: `0` = WT / first-round panel; `1` = round 1; `2` = round 2; etc. |

---

### `this_round_variants.csv`

The subset of variants actually used to train the model in this call — i.e. everything
in `iteration.csv` that was passed as `iter_train` to `top_layer`. In practice this is
all measured variants from all completed rounds.

| Column    | Type | Description |
|-----------|------|-------------|
| `variant` | str  | Variant name of each training-set member |

---

### `df_test.csv` ⭐ primary output

Predictions for every variant that has **not yet been measured**. Sorted by `y_pred`
descending — the top rows are the model's recommended candidates for the next assay round.

| Column           | Type  | Description |
|------------------|-------|-------------|
| `variant`        | str   | Variant name |
| `y_pred`         | float | Predicted activity from the regression model. Higher = model expects better activity. **Use this column to rank candidates.** |
| `y_actual`       | float | Always `NaN` in experimental mode (no ground truth yet). |
| `y_actual_scaled`| float | Always `NaN` in experimental mode. |
| `y_actual_binary`| float | Always `NaN` in experimental mode. |
| `dist_metric`    | float | Always `NaN` in experimental mode (populated in simulation mode only). |
| `std_predictions`| float | `0.0` for all models — reserved for future uncertainty-aware strategies. |

> **Note:** `y_actual`, `y_actual_scaled`, `y_actual_binary`, and `dist_metric` are
> always `NaN` in experimental mode because these variants have no ground-truth
> measurements yet. They are included so the file has the same schema as `df_sorted_all`.

---

### `df_sorted_all.csv`

All variants — both the training set and untested variants — ranked together by `y_pred`
descending. Useful for seeing where each tested variant falls in the model's overall
ranking.

| Column           | Type  | Training-set rows | Untested rows |
|------------------|-------|-------------------|---------------|
| `variant`        | str   | Variant name | Variant name |
| `y_pred`         | float | Model's back-prediction on training data | Model's forward prediction |
| `y_actual`       | float | Measured activity value | `NaN` |
| `y_actual_scaled`| float | Min-max scaled measured activity (0–1) | `NaN` |
| `y_actual_binary`| float | `1.0` if activity ≥ 1, else `0.0` | `NaN` |
| `dist_metric`    | float | Min Euclidean distance from this training variant to any untested variant | `NaN` (experimental mode) |
| `std_predictions`| float | `0.0` (reserved) | `0.0` (reserved) |

---

## Multi-mutant workflow (`evolve_experimental_multi`)

### When to use it

Use `evolve_experimental_multi()` instead of `evolve_experimental()` once your campaign
moves into combinatorial mutagenesis — i.e. when any assay round includes variants with
mutations at **more than one position** (e.g. `A107M_F73C`). Single-mutant-only
campaigns should continue to use `evolve_experimental()`.

### Key differences from `evolve_experimental`

| | `evolve_experimental` | `evolve_experimental_multi` |
|---|---|---|
| Variant types supported | Single-mutant only | Single-mutant **and** multi-mutant |
| Embeddings input | One CSV file (`embeddings_file`) | List of CSV files (`embeddings_files`) |
| Round file inputs | One list (`round_files`) | Two lists: `round_files_single` and `round_files_multi` |
| Output subdirectory | `<output_dir>/<round_name>/` | `<output_dir>/<round_name>/` |
| Return value | None (saves files only) | None (saves files only) |

### How embeddings are handled

`evolve_experimental_multi` accepts a list of embeddings CSV files (`embeddings_files`).
The files are concatenated in order, with one important rule: WT is retained from the
**first** file only — any row named `"WT"` in subsequent files is dropped automatically
(with a warning) to avoid duplicate index entries. An error is raised if embedding files
have inconsistent numbers of dimensions or if duplicate variant names appear after
concatenation.

### How round files are handled

Round files are split into two separate lists:

- `round_files_single` — Excel files containing single-mutant assay results. Variant
  names must be in full standard notation (e.g. `"A107M"`).
- `round_files_multi` — Excel files containing multi-mutant assay results (e.g.
  `"A107M_F73C"`).

Both lists are concatenated and passed to `create_iteration_dataframes` in order: all
single-mutant rounds first, then all multi-mutant rounds. Iteration numbers are assigned
chronologically across both lists, so the first single-mutant file is round 1, the
second is round 2, and so on, with multi-mutant files continuing the sequence.

### Example usage

```python
from evolvepro.src.evolve import evolve_experimental_multi

base_path = "/path/to/D4/"

evolve_experimental_multi(
    round_name='multi_1',
    embeddings_files=[
        base_path + 'embeddings/mutants_esm2_t33_650M_UR50D.csv',       # single-mutant (contains WT)
        base_path + 'embeddings/multi_mutants_esm2_t33_650M_UR50D.csv', # multi-mutant (no WT row)
    ],
    round_files_single=[
        base_path + 'experimental_data/D4_round_1.xlsx',
        base_path + 'experimental_data/D4_round_2.xlsx',
    ],
    round_files_multi=[
        base_path + 'experimental_data/D4_multi_round_1.xlsx',
    ],
    wt_fasta_path=base_path + 'WT.fasta',
    output_dir=base_path + 'evolve_outputs/',
)
```

### Output files

`evolve_experimental_multi` saves output under `<output_dir>/<round_name>/`, the same
structure as `evolve_experimental`:

| File                      | Contents |
|---------------------------|----------|
| `iteration.csv`           | Which variants were tested in which round |
| `this_round_variants.csv` | Variants used to train the model this round |
| `df_test.csv`             | All untested variants ranked by predicted activity |
| `df_sorted_all.csv`       | All variants (train + test) sorted by predicted activity |

---

## Module reference

### `data.py`

**`validate_csv(file_path, file_type)`**
Loads and validates a CSV (or Excel) file. `file_type` must be `"experimental"` or
`"embeddings"`. Experimental files are read with `pd.read_excel`; embedding files with
`pd.read_csv`. Raises `FileNotFoundError` or `ValueError` on any structural problem.

**`load_experimental_embeddings(file)`**
Thin wrapper around `validate_csv(..., 'embeddings')`. Returns the embeddings DataFrame
with the variant column as the index.

**`load_experimental_data(file, wt_fasta_path)`**
Thin wrapper around `validate_csv(..., 'experimental')`. Returns the round DataFrame
with lowercase column names.

**`create_iteration_dataframes(df_list, expected_variants)`**
Combines multiple round DataFrames into `iteration` and `labels` DataFrames. Assigns
iteration 0 to WT (round 1 only) and assigns each subsequent round a 1-based integer.
Raises `ValueError` on duplicate variants. Appends placeholder rows (all `NaN`) for
variants in `expected_variants` that have not yet been tested.

---

### `model.py`

**`first_round(labels, embeddings, ...)`**
Selects the initial panel of variants to test before any measurements exist. Three
strategies are available:

- `"random"` — uniform random sampling (excluding WT).
- `"diverse_medoids"` — K-medoids clustering in embedding space; returns the medoid
  of each cluster. Requires `scikit-learn-extra`.
- `"explicit_variants"` — caller-supplied list of variant names.

Returns `(labels_zero, iteration_zero, this_round_variants)`.

**`top_layer(iter_train, iter_test, embeddings_pd, labels_pd, measured_var, ...)`**
Trains a regression model on the variants in `iter_train` and predicts activity for
the test set. In experimental mode (`experimental=True`), the test set is all variants
with `NaN` iteration, and the function returns `(this_round_variants, df_test, df_sorted_all)`.
In simulation mode, it returns a tuple of metrics plus `df_test` and `this_round_variants`.

Supported `regression_type` values: `"ridge"`, `"lasso"`, `"elasticnet"`, `"linear"`,
`"neuralnet"`, `"randomforest"` (default), `"gradientboosting"`, `"knn"`, `"gp"`.

---

### `process.py`

**DMS pre-processing:**

- **`process_dataset(...)`** — reads a raw DMS Excel/CSV, applies activity cutoffs,
  writes `<dataset_name>_labels.csv` and `<dataset_name>.fasta`.
- **`generate_mutation_fasta(...)`** — helper called by `process_dataset`; writes a
  FASTA of mutant sequences from a variant DataFrame.
- **`markin_custom_cutoff(...)`** — custom binary-label rule for the Markin et al. dataset.
- **`preprocess_cas12f(...)`** — dataset-specific pre-processor for Cas12f raw data.
- **`preprocess_cov2_S(...)`** — dataset-specific pre-processor for SARS-CoV-2 spike data.

**Experimental setup:**

- **`generate_wt(wt_sequence, output_file)`** — writes a single-record FASTA with
  record ID `WT`.
- **`generate_single_aa_mutants(wt_fasta, output_file, positions=None)`** — generates
  all single amino-acid substitutions (or a position-restricted subset) and writes a
  FASTA. The WT sequence is included as the first record.
- **`generate_n_mutant_combinations(wt_fasta, mutant_file, n, output_file, threshold=1.0)`**
  — reads beneficial single mutants from an Excel file (position-only format), generates
  all valid n-fold combinations, and writes a FASTA. Combinations mutating the same
  position twice are skipped.
- **`suggest_initial_mutants(fasta_file, num_mutants=10, random_seed=None)`** — randomly
  samples an initial panel of variant names from a FASTA file. For embedding-space-aware
  selection use `first_round(..., first_round_strategy="diverse_medoids")` instead.

**DMS exploration plots:**

- **`plot_mutations_per_position(df)`** — bar chart of mutation counts per sequence position.
- **`plot_histogram_of_readout(df, column_name, cutoff=None)`** — histogram of activity
  values with optional WT and cutoff markers.

---

### `plot.py`

Plotting utilities for experimental campaign results.

**`plot_y_pred_distribution(df, ...)`**
Histogram of predicted fitness scores (`y_pred`) for all variants. Can be stratified
by mutation count (`by_n_mutations=True`) and optionally highlights the WT prediction
with a dashed vertical line (`highlight_wt=True`).

**`plot_additive_vs_actual(df, ...)`**
Scatter plot comparing the additive predicted score (sum of constituent single-mutant
`y_pred` values) against the actual model prediction for each multi-mutant variant.
Useful for visualising epistasis. Only multi-mutants whose component single mutations
all appear in the DataFrame are included; others are silently skipped.

---

### `plot_dms.py`

Plotting utilities for DMS simulation results.

**`read_dms_data(directory, datasets, model, experiment, group_columns, aggregate_columns, ...)`**
Reads and aggregates simulation result CSVs across multiple datasets. Each CSV is
expected to follow the naming convention `{dataset}_{model}_{experiment}.csv`. Returns
a concatenated DataFrame with per-group mean and std columns for each aggregate column.

---

### `utils.py`

**`pca_embeddings(embeddings_df, n_components=10)`**
Reduces an embeddings DataFrame to its top principal components. Returns a new DataFrame
with columns `"PCA 1"`, `"PCA 2"`, …, `"PCA <n_components>"` and the original variant
name index. Raises `ValueError` if the DataFrame is empty, if `n_components` is invalid,
or if any non-numeric columns are present.
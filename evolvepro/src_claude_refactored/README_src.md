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
├── data.py      — file loading and alignment helpers
├── model.py     — first-round selection and active-learning model
├── evolve.py    — top-level orchestration (simulation + experimental workflows)
├── process.py   — pre-processing raw DMS / experimental data; FASTA generation
├── plot.py      — plotting and reporting utilities
└── utils.py     — shared helpers (PCA wrapper)
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
- Used to reconstruct mutant sequences from variant names, and to generate a WT entry
  inside experimental round files.

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
- Used by `suggest_initial_mutants()` to sample a diverse starting panel.

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
  round files and the labels file.
- All values must be numeric; no missing entries are allowed.
- Embeddings are produced by running a protein language model on `mutants.fasta` using
  one of the `extract_*.py` scripts (see `README_plm.md`). Every script outputs a CSV
  in this exact format.

Alternatively, embeddings can be stored as a PyTorch `.pt` file — a dictionary mapping
variant names to tensors with keys `"average"` and/or `"mutated"`. This format is
supported only in DMS/simulation mode (`load_dms_data`); the experimental functions
require a CSV.

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
index of the output CSV will already be in the correct format.

**Step 3 — Check the WT row name.**
Most PLMs use the full FASTA header as the record ID. If the WT record in your FASTA
is `>WT Wild-type sequence`, the resulting CSV will have a row named
`"WT Wild-type sequence"` rather than `"WT"`. Pass `rename_WT=True` to
`evolve_experimental()` to fix this automatically:

```python
evolve_experimental(
    ...
    embeddings_file_name='mutants_esm2_t33_650M_UR50D.csv',
    rename_WT=True,   # renames "WT Wild-type sequence" → "WT"
)
```

If your WT record is already `>WT` in the FASTA, leave `rename_WT=False` (the default).

**Step 4 — Verify coverage.**
Every variant name that appears in any of your round Excel files must have a
corresponding row in the embeddings CSV, and conversely every candidate you might
want recommended must also have a row (the model can only score variants it has an
embedding for). Variants present in the embeddings but absent from round files are
treated as the unlabelled prediction pool — this is expected and correct.

**Quick checklist for `evolve_experimental`:**

| Check | What to verify |
|---|---|
| Single CSV | One file covering WT + all single-mutant candidates |
| Row index = variant name | e.g. `WT`, `A107M`, `F73C` — not FASTA descriptions |
| WT row present | Required; rename with `rename_WT=True` if necessary |
| No missing values | All embedding dimensions must be numeric |
| Coverage | Every tested and candidate variant has a row |

---

#### Preparing embeddings for `evolve_experimental_multi()`

`evolve_experimental_multi` accepts a **list** of embeddings CSV files
(`embeddings_file_names`). This is necessary because single-mutant and multi-mutant
libraries are typically embedded separately — the multi-mutant FASTA is only generated
after the single-mutant rounds have identified the best positions to combine.

The files are concatenated in the order they appear in the list. One important
constraint: **WT must appear in exactly one file** (the first). Any row named `"WT"` or
`"WT Wild-type sequence"` in the second or later files is silently dropped before
concatenation to prevent duplicate index entries. The `rename_WT` flag applies only to
the first file.

**Typical setup — two embeddings files:**

```
embeddings_file_names=[
    'mutants_esm2_t33_650M_UR50D.csv',        # single-mutant library — MUST contain WT
    'multi_mutants_esm2_t33_650M_UR50D.csv',  # multi-mutant library — must NOT contain WT
]
```

**Step 1 — Embed the single-mutant library (same as `evolve_experimental`).**
Follow steps 1–3 above for `mutants.fasta`. This file should contain WT and all
single-mutant candidates. Use this as the **first** entry in `embeddings_file_names`.

**Step 2 — Generate and embed the multi-mutant library.**
Once you have identified promising single mutations, generate a combinatorial FASTA
(e.g. with `generate_n_mutant_combinations()` from `process.py`) containing all
multi-mutant candidates you want to evaluate. **Do not include WT** in this FASTA —
if it ends up in the CSV anyway, it will be dropped automatically, but omitting it
avoids unnecessary computation.

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
(1024 dims) will cause a silent shape mismatch and incorrect results.

**Step 4 — Ensure row index format matches round file variant names.**
Multi-mutant variant names must use underscore-joined standard notation, e.g.
`A107M_F73C`. The record IDs in your multi-mutant FASTA must already be in this
format, so that the row index of the CSV matches the `Variant` column in the
multi-mutant round Excel files (which are used as-is, with no name conversion applied).

**Quick checklist for `evolve_experimental_multi`:**

| Check | What to verify |
|---|---|
| Same PLM + layer for all files | Ensures consistent embedding dimensionality |
| WT in first file only | WT rows in subsequent files are dropped automatically |
| `rename_WT` applies to first file only | Subsequent files are not renamed |
| Multi-mutant row index = full variant name | e.g. `A107M_F73C`, not `107M_73C` |
| No missing values in any file | All embedding dimensions must be numeric |
| Coverage | Every candidate (single- and multi-mutant) has a row in one of the files |

---

### 4. Experimental round file (`D4_round_N.xlsx`)

An Excel workbook (`.xlsx`) with **at least** these two columns (additional columns are
ignored unless `drop_columns=False`):

| Column     | Type   | Description |
|------------|--------|-------------|
| `Variant`  | string | Variant identifier (see §5 below) |
| `activity` | float  | Measured activity value for that variant |

**Rules:**
- One row per tested variant.
- WT must appear in round 1 as `Variant = "WT"`.
- WT should **not** be re-listed in subsequent rounds.
- No duplicate variants across all rounds (the code will abort and print duplicates if
  found).
- Missing or blank `activity` values are dropped before processing.

**Single-mutant variant column format (when `single_mutant=True`):**

The `Variant` column uses a *position-only* format that the code converts to full
standard notation. Given a WT sequence:

```
Variant column value   →   Processed variant name
"107M"                 →   "A107M"   (A is the WT amino acid at position 107)
"WT"                   →   "WT"
```

That is: the code reads the numeric position, looks up the WT amino acid from the FASTA,
and prepends it. Do **not** include the WT amino acid in the Excel `Variant` column for
single-mutant mode.

**Multi-mutant variant column format (when `single_mutant=False`):**

The `Variant` column must already contain the complete variant name, e.g. `"A107M_F73C"`.
No conversion is applied. Pass these files to `round_file_names_multi` in
`evolve_experimental_multi()`.

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

Produced by `process_dataset()` and consumed by `load_dms_data()`.

| Column           | Type   | Description |
|------------------|--------|-------------|
| `variant`        | string | Variant name in standard notation |
| `activity`       | float  | Raw activity value |
| `activity_scaled`| float  | Min-max scaled activity (0–1) |
| `activity_binary`| int    | 1 if activity exceeds the chosen cutoff, else 0 |

Additional `activity_binary_<p>p` columns may be present if percentile cutoffs were
requested.

---

## Typical experimental workflow

```
1. generate_wt()                    →  WT.fasta
2. generate_single_aa_mutants()     →  mutants.fasta
3. <Run ESM / language model>       →  mutants_esm2_t33_650M_UR50D.csv
4. suggest_initial_mutants()        →  pick first 36 variants to test
5. <Run wet-lab assay>              →  D4_round_1.xlsx
6. evolve_experimental()            →  ranked predictions for next round
7. <Run wet-lab assay>              →  D4_round_2.xlsx
8. evolve_experimental()            →  ... repeat
```

---

## Output files (written by `evolve_experimental`)

Each call to `evolve_experimental()` creates a subdirectory `<output_dir>/<round_name>/`
containing four CSV files.  The primary file for planning the next round is
**`df_test.csv`** — sort by `y_pred` descending and pick from the top.

---

### `iteration.csv`

Records which assay round each variant was tested in.  Contains only variants that have
been measured (i.e. the training set); untested variants are not listed.

| Column      | Type  | Description |
|-------------|-------|-------------|
| `variant`   | str   | Variant name in standard notation (e.g. `A107M`) |
| `iteration` | float | Round number: `0` = WT / first-round panel; `1` = round 1; `2` = round 2; etc. |

---

### `this_round_variants.csv`

The subset of variants actually used to train the model in this call — i.e. everything
in `iteration.csv` that was passed as `iter_train` to `top_layer`.  In practice this is
all measured variants from all completed rounds.

| Column    | Type | Description |
|-----------|------|-------------|
| `variant` | str  | Variant name of each training-set member |

---

### `df_test.csv` ⭐ primary output

Predictions for every variant that has **not yet been measured**.  Sorted by `y_pred`
descending — the top rows are the model's recommended candidates for the next assay round.

| Column           | Type  | Description |
|------------------|-------|-------------|
| `variant`        | str   | Variant name |
| `y_pred`         | float | Predicted activity from the regression model.  Higher = model expects better activity.  **Use this column to rank candidates.** |
| `y_actual`       | float | Measured activity value.  Always `NaN` for untested variants in experimental mode. |
| `y_actual_scaled`| float | Min-max scaled measured activity (0–1, where 1 = best observed so far).  Always `NaN` for untested variants in experimental mode. |
| `y_actual_binary`| float | `1.0` if measured activity ≥ 1, `0.0` otherwise.  Always `NaN` for untested variants in experimental mode. |
| `dist_metric`    | float | Always `NaN` in experimental mode (populated in simulation mode only). |
| `std_predictions`| float | Prediction uncertainty.  Currently `0.0` for all models — reserved for future uncertainty-aware strategies. |

> **Note:** `y_actual`, `y_actual_scaled`, `y_actual_binary`, and `dist_metric` are
> always `NaN` in experimental mode because these variants have no ground-truth
> measurements yet.  They are included so the file has the same schema as `df_sorted_all`.

---

### `df_sorted_all.csv`

All variants — both the training set and untested variants — ranked together by `y_pred`
descending.  Useful for seeing where each tested variant falls in the model's overall
ranking, and for identifying any training-set variants the model would rank highly.

Contains the same columns as `df_test.csv`, with the distinction that training-set rows
have real values in `y_actual`, `y_actual_scaled`, and `y_actual_binary`, while untested
rows have `NaN` in those columns.

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
mutations at **more than one position** (e.g. `A107M_F73C`).  Single-mutant-only
campaigns should continue to use `evolve_experimental()`.

### Key differences from `evolve_experimental`

| | `evolve_experimental` | `evolve_experimental_multi` |
|---|---|---|
| Variant types supported | Single-mutant only | Single-mutant **and** multi-mutant |
| Embeddings input | One CSV file (`embeddings_file_name`) | One or more CSV files (`embeddings_file_names` list) |
| Round file inputs | One list (`round_file_names`) | Two lists: `round_file_names_single` and `round_file_names_multi` |
| Variant name conversion | Auto-converts position-only format (e.g. `"107M"` → `"A107M"`) | Single-mutant files are converted; multi-mutant files are used as-is |
| Output subdirectory | `<output_dir>/<round_name>/` | `<output_dir>/<protein_name>/<round_name>/` |
| Return value | None (saves files only) | Returns `(this_round_variants, df_test, df_sorted_all)` |

### How embeddings are handled

`evolve_experimental_multi` accepts a **list** of embeddings CSV files
(`embeddings_file_names`).  This is necessary because single-mutant and multi-mutant
libraries are typically embedded separately.  The files are concatenated in order, with
one important rule: WT is retained from the **first** file only — any row named `"WT"`
or `"WT Wild-type sequence"` in subsequent files is dropped automatically to avoid
duplicates.

The `rename_WT` flag applies only to the first embeddings file.

### How round files are handled

Round files are split into two separate lists:

- `round_file_names_single` — Excel files where the `Variant` column uses the
  position-only format (e.g. `"107M"`).  Processed identically to `evolve_experimental`.
- `round_file_names_multi` — Excel files where the `Variant` column already contains
  fully specified multi-mutant names (e.g. `"A107M_F73C"`).  No name conversion is
  applied.

Both lists are passed to `create_iteration_dataframes` in order: all single-mutant rounds
first, then all multi-mutant rounds.  Iteration numbers are assigned chronologically
across both lists, so the first single-mutant file is round 1, the second is round 2,
and so on, with multi-mutant files continuing the sequence.

### Example usage

```python
from evolvepro.src.evolve import evolve_experimental_multi

base_path = "/path/to/D4/"

this_round_variants, df_test, df_sorted_all = evolve_experimental_multi(
    protein_name='D4',
    round_name='multi_1',
    embeddings_base_path=base_path + 'embeddings/',
    embeddings_file_names=[
        'mutants_esm2_t33_650M_UR50D.csv',       # single-mutant library (contains WT)
        'multi_mutants_esm2_t33_650M_UR50D.csv',  # multi-mutant library (no WT row)
    ],
    round_base_path=base_path + 'experimental_data/',
    round_file_names_single=[
        'D4_round_1.xlsx',   # single-mutant assay results
        'D4_round_2.xlsx',
    ],
    round_file_names_multi=[
        'D4_multi_round_1.xlsx',  # multi-mutant assay results (full variant names)
    ],
    wt_fasta_path=base_path + 'WT.fasta',
    rename_WT=True,
    number_of_variants=32,
    output_dir=base_path + 'evolve_outputs/',
)
```

### Output files

`evolve_experimental_multi` saves output under `<output_dir>/<protein_name>/<round_name>/`
(note the extra `<protein_name>` level compared to `evolve_experimental`):

| File                      | Contents |
|---------------------------|----------|
| `iteration.csv`           | Which variants were tested in which round |
| `this_round_variants.csv` | Variants used to train the model this round |
| `df_test.csv`             | All untested variants ranked by predicted activity |
| `df_sorted_all.csv`       | All variants (train + test) sorted by predicted activity |

The function also **returns** `(this_round_variants, df_test, df_sorted_all)` directly,
unlike `evolve_experimental` which only saves to disk.  The primary file for planning
the next round remains **`df_test.csv`** — top rows are the model's recommendations.
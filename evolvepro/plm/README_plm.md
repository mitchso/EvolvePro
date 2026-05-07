# Protein Language Model Embedding Extraction

This repository contains scripts for extracting protein sequence embeddings from several protein language models (PLMs). Each script takes a FASTA file as input and outputs a CSV file where each row corresponds to a sequence and each column corresponds to a dimension of the embedding vector.

---

## Overview of Available Models

| Script | Model | Embedding Dim | GPU Required | Notes |
|---|---|---|---|---|
| `extract_esm.py` | ESM-2 / ESM-1b (Meta) | Varies by model | Recommended | Most flexible; supports per-token and mean representations |
| `extract_prot_t5.py` | ProtT5-XL (Rostlab) | 1024 | Recommended | Strong general-purpose embeddings |
| `extract_ankh.py` | Ankh (ElnaggarLab) | 1536 (large) / 768 (base) | Recommended | Modern; good for diverse tasks |
| `extract_proteinbert.py` | ProteinBERT | 1562 | Optional | Requires cloning external repo |
| `extract_unirep.py` | UniRep | 1900 | No (JAX) | LSTM-based; no transformer |
| `extract_one-hot.py` | One-hot / Integer | 20 × seq_len | No | Baseline encoding; no learned representations |

---

## Setup

### 1. Install dependencies

Each model has its own dependencies. Install only what you need:

```bash
# ESM
pip install fair-esm

# ProtT5
pip install transformers sentencepiece

# Ankh
pip install ankh

# ProteinBERT + UniRep (requires cloning external repos first)
python clone_git_plms.py

# UniRep
pip install jax-unirep

# Shared utilities
pip install biopython pandas numpy torch
```

### 2. Clone external repositories (ProteinBERT only)

```bash
python clone_git_plms.py
```

This clones the `protein_bert` and `efficient-evolution` repositories into an `external/` directory and installs ProteinBERT.

---

## Usage

### ESM (`extract_esm.py`)

Extracts embeddings from Meta's ESM family of models.

```bash
python extract_esm.py <model_name> <fasta_file> <output_dir> \
    --include mean \
    --repr_layers -1 \
    --concatenate_dir <csv_output_dir>
```

#### Arguments

**`model_location`** (positional, required)

The pretrained ESM model to use. Can be either:
- A named model string (e.g. `esm2_t33_650M_UR50D`, `esm1b_t33_650M_UR50S`) — the model weights are downloaded automatically on first use and cached locally.
- A path to a local `.pt` file if you have already downloaded model weights.

See the ESM-2 Model Variants table below for a full list of available named models and their trade-offs.

---

**`fasta_file`** (positional, required)

Path to the input FASTA file containing the protein sequences to embed. Each sequence header (the `>name` line) is used as the row index in the output CSV.

---

**`output_dir`** (positional, required)

Directory where intermediate per-sequence `.pt` files are saved during extraction. Each sequence is saved as a separate PyTorch tensor file named `<sequence_id>.pt`. These files store the raw representations before any aggregation into a CSV.

This directory is created automatically if it does not exist. If you also pass `--concatenate_dir`, these `.pt` files are temporary and can be deleted after the CSV is produced. If you do not pass `--concatenate_dir`, these `.pt` files are your final output and must be loaded individually with `torch.load()`.

---

**`--include`** (required, one or more of: `mean`, `per_tok`, `bos`, `contacts`)

Controls what type of representation is extracted from the model and saved into each `.pt` file. You can specify one or more options at a time. This is the most consequential argument for downstream use.

- **`mean`** — Computes the average of all per-residue token embeddings across the full sequence (excluding special tokens), producing a single fixed-length vector per sequence regardless of sequence length. This is the most common choice for protein-level prediction tasks such as fitness prediction, functional annotation, or classification. It is the right default for most users.

- **`per_tok`** — Saves the full matrix of per-residue embeddings, one vector per amino acid position. The output shape is `(sequence_length, embedding_dim)`. Use this when your downstream task needs positional information — for example, residue-level mutation effect scoring, contact prediction fine-tuning, or any model that consumes a sequence of embeddings rather than a single pooled vector. Be aware that sequences of different lengths will produce tensors of different shapes, which requires padding or masking before batched training.

- **`bos`** — Saves only the embedding at the first position (the `[CLS]` / beginning-of-sequence token). This is analogous to the `[CLS]` token representation used in BERT-style models and captures a global sequence summary in a single vector. In practice `mean` tends to outperform `bos` for most protein tasks, but `bos` may be worth exploring if you are reproducing a specific protocol or comparing to a baseline that used it.

- **`contacts`** — Returns a predicted residue–residue contact map (an `L × L` matrix, where `L` is sequence length) derived from the model's attention heads. This is not an embedding in the traditional sense; it is a structural prediction output. Use this if you need predicted contact maps directly, not for featurising sequences for a downstream model.

You can specify multiple options simultaneously (e.g. `--include mean per_tok`) and all selected representations will be stored in each `.pt` file. Note that `--concatenate_dir` (see below) only reads the `mean` representation when assembling the CSV, so if you intend to use the concatenation step, `mean` must be among the included options.

---

**`--repr_layers`** (default: `-1`)

Selects which transformer layer(s) to extract representations from. Layers are indexed from `0` (the embedding layer) to `num_layers` (the final layer). Negative indices count from the end, so `-1` (the default) refers to the last layer.

In most cases the final layer (`-1`) produces the best embeddings for fitness and function prediction tasks. For structure-sensitive applications — for example, predicting secondary structure or contact maps — representations from intermediate layers (roughly the last quarter of the network) can carry complementary geometric information and are worth exploring. You can extract from multiple layers simultaneously by passing several values (e.g. `--repr_layers -1 -4 -8`), in which case each layer is stored separately in the `.pt` file.

---

**`--toks_per_batch`** (default: `4096`)

Controls the maximum number of amino acid tokens processed in a single forward pass. Sequences are automatically grouped into batches that stay within this budget. Increasing this value improves GPU utilisation and speeds up extraction but consumes more VRAM. Decrease it if you encounter out-of-memory errors.

As a rough guide: on a 16 GB GPU with `esm2_t33_650M_UR50D`, the default of 4096 is safe for most datasets. With the 3B model, you may need to reduce to 1024–2048.

---

**`--truncation_seq_length`** (default: `1022`)

Sequences longer than this value are truncated before embedding. The ESM-2 models have a hard positional encoding limit of 1022 residues (not including the two special tokens), so this is also the effective maximum regardless of what value is set. If your dataset contains very long sequences (e.g. multi-domain proteins), consider tiling the sequence into overlapping windows and aggregating the resulting embeddings, or using an alternative model without this length constraint.

---

**`--concatenate_dir`** (default: not set)

If provided, this argument triggers a second step after extraction: all per-sequence `.pt` files in `output_dir` are loaded, their `mean` representations are extracted, and the results are concatenated into a single CSV file. The CSV is saved to `<concatenate_dir>/<fasta_stem>_<model_name>.csv`, with one row per sequence (indexed by sequence ID) and one column per embedding dimension.

Without this flag, the script stops after saving the individual `.pt` files and no CSV is produced. In that case, you would need to load and aggregate the `.pt` files yourself using `torch.load()`.

In short: **if you want a ready-to-use CSV of embeddings, always pass `--concatenate_dir`.** The intermediate `.pt` files in `output_dir` can then be deleted once the CSV is confirmed.

---

**`--nogpu`** (flag, default: off)

Forces the model to run on CPU even if a GPU is available. Useful for debugging or on machines without a compatible GPU. For large models or datasets, CPU inference is significantly slower.

---

#### Two-step pipeline

It may help to think of `extract_esm.py` as a two-step process:

1. **Extraction** (`run()`): reads the FASTA file, runs sequences through the model in batches, and writes one `.pt` file per sequence into `output_dir`. Each `.pt` file is a dictionary containing the label and whichever representation types were requested via `--include`.

2. **Concatenation** (`concatenate_files()`): triggered only when `--concatenate_dir` is set. Walks `output_dir`, loads every `.pt` file, reads the `mean_representations` entry, and assembles a single pandas DataFrame that is saved as a CSV. This step requires that `mean` was included in `--include`.

---

**Example — recommended default (650M model, mean embeddings, CSV output):**
```bash
python extract_esm.py esm2_t33_650M_UR50D sequences.fasta esm_out/ \
    --include mean \
    --repr_layers -1 \
    --concatenate_dir results/
```

**Example — per-residue embeddings for a structure-sensitive task:**
```bash
python extract_esm.py esm2_t33_650M_UR50D sequences.fasta esm_out/ \
    --include per_tok \
    --repr_layers -1
```
Note: no `--concatenate_dir` here because the concatenation step only handles `mean` representations.

**Example — lightweight 150M model for CPU or low-memory environments:**
```bash
python extract_esm.py esm2_t30_150M_UR50D sequences.fasta esm_out/ \
    --include mean \
    --repr_layers -1 \
    --concatenate_dir results/
```

---

### ESM-2 Model Variants — Performance and Trade-offs

ESM-2 is available in several sizes. The model name passed as `model_location` determines the architecture used. All variants were trained on UniRef50 and share the same general API, but differ substantially in embedding quality, memory footprint, and runtime.

| Model name | Parameters | Layers | Embedding dim | Approx. VRAM | Relative quality |
|---|---|---|---|---|---|
| `esm2_t6_8M_UR50D` | 8M | 6 | 320 | < 1 GB | Baseline |
| `esm2_t12_35M_UR50D` | 35M | 12 | 480 | ~1 GB | Low |
| `esm2_t30_150M_UR50D` | 150M | 30 | 640 | ~2 GB | Moderate |
| `esm2_t33_650M_UR50D` | 650M | 33 | 1280 | ~4 GB | **Good — recommended default** |
| `esm2_t36_3B_UR50D` | 3B | 36 | 2560 | ~12 GB | High |
| `esm2_t48_15B_UR50D` | 15B | 48 | 5120 | ~60 GB | Highest |

**Key considerations when choosing a variant:**

**Embedding quality and downstream task performance.** Larger models consistently produce richer representations. In benchmarks covering structure prediction, function annotation, and fitness prediction tasks, the 650M model (`t33`) offers strong performance for most protein engineering applications with manageable compute costs. The 3B model (`t36`) provides meaningful gains if your downstream task is sensitive to subtle sequence features or you are working with structurally complex or highly divergent protein families. The 15B model offers marginal further improvement over 3B for most tasks and is rarely worth the infrastructure cost unless you have a specific high-precision requirement.

**Sequence length.** All ESM-2 variants have a maximum context of 1022 residues (set by `--truncation_seq_length`). This limit is the same regardless of model size, so larger models do not help with very long sequences — consider tiling or alternative models in that case.

**Layer selection.** The `--repr_layers` argument controls which transformer layers are used to generate embeddings. The final layer (`-1`, the default) generally performs best for fitness and function prediction. For structure-sensitive tasks, intermediate layers (roughly the last quarter of the network) have been shown to carry complementary geometric information and can be worth exploring.

**Compute budget guidance.** If you are running on a single consumer GPU (≤ 16 GB VRAM), `esm2_t33_650M_UR50D` is the practical ceiling. If only a CPU is available, the 8M or 35M variants are usable, though slow for large datasets. For high-throughput screening campaigns with many variants, prefer the 150M or 650M model and increase `--toks_per_batch` to maximise GPU utilisation.

**Reproducibility.** If you are reproducing results from a published study, check which ESM-2 variant was used in that work. Switching model sizes changes embedding dimensionality and representation geometry, which will affect any downstream model trained on those embeddings.

---

### ProtT5 (`extract_prot_t5.py`)

Extracts embeddings from the ProtT5-XL model, trained on UniRef50.

```bash
python extract_prot_t5.py -i sequences.fasta -o embeddings.csv \
    --per_protein 1
```

**Arguments:**
- `-i` / `--input` — Input FASTA file
- `-o` / `--output` — Output CSV file path
- `--model` — Path to a locally cached model directory (optional; downloads automatically if not set)
- `--per_protein` — `1` for mean-pooled per-protein embeddings, `0` for per-residue (default: `0`)

**Notes:**
- Non-standard amino acids (`U`, `Z`, `O`) are replaced with `X` automatically.
- Model weights (~3 GB) are downloaded and cached on first run.

---

### Ankh (`extract_ankh.py`)

Extracts mean-pooled embeddings from the Ankh model (base or large).

```bash
python extract_ankh.py -i sequences.fasta -o embeddings.csv \
    --model large \
    --batch_size 50
```

**Arguments:**
- `-i` / `--input` — Input FASTA file
- `-o` / `--output` — Output CSV file path
- `--model` — Model size: `large` (default) or `base`
- `--batch_size` — Sequences per batch (default: 50; reduce if running out of memory)

---

### ProteinBERT (`extract_proteinbert.py`)

Extracts global embeddings from ProteinBERT.

```bash
python extract_proteinbert.py -i sequences.fasta -o embeddings.csv
```

**Arguments:**
- `-i` / `--input` — Input FASTA file
- `-o` / `--output` — Output CSV file path

**Notes:**
- Requires running `clone_git_plms.py` first to set up the external repository.
- Model weights are downloaded automatically on first run.
- Sequence length is automatically padded to 512 or `max_seq_len + 2`, whichever is larger.

---

### UniRep (`extract_unirep.py`)

Extracts mean-hidden-state embeddings using the JAX-based UniRep implementation.

```bash
python extract_unirep.py -i sequences.fasta -o embeddings.csv
```

**Arguments:**
- `-i` / `--input` — Input FASTA file
- `-o` / `--output` — Output CSV file path

**Notes:**
- Uses the `h_avg` (average hidden state) representation.
- Runs on CPU via JAX; no GPU setup needed.
- Requires `jax-unirep` to be installed.

---

### One-Hot / Integer Encoding (`extract_one-hot.py`)

Encodes sequences as fixed-length one-hot or integer vectors. This is a simple baseline with no learned representations.

```bash
python extract_one-hot.py sequences.fasta \
    --method one_hot \
    --results_path results/
```

**Arguments:**
- `fasta_file` — Input FASTA file (positional)
- `--method` — Encoding method: `one_hot` or `integer` (required)
- `--results_path` — Directory to save output CSV (default: `results/`)
- `--verbose` — Print each sequence ID as it is processed

**Notes:**
- One-hot encoding produces a vector of length `20 × sequence_length`. Sequences of different lengths will produce vectors of different lengths, which may need to be handled before downstream use.
- Only the 20 standard amino acids are encoded; non-standard residues are ignored.

---

## Choosing a Model

Use this decision guide to select the right embedding script for your task:

**Use `extract_esm.py`** if you want state-of-the-art embeddings with the most flexibility. ESM-2 models are among the best-benchmarked PLMs and support a range of model sizes to suit your compute budget.

**Use `extract_prot_t5.py`** if you need strong general-purpose embeddings and have a GPU available. ProtT5 performs excellently on structure and function prediction tasks.

**Use `extract_ankh.py`** if you want a modern alternative to ESM/ProtT5 with competitive performance and a simpler API.

**Use `extract_proteinbert.py`** if you specifically need ProteinBERT embeddings or are reproducing prior work that used this model.

**Use `extract_unirep.py`** if you need an LSTM-based embedding (e.g. to reproduce older baselines) or if you do not have access to a GPU.

**Use `extract_one-hot.py`** if you need a simple, interpretable baseline or are testing a pipeline before committing to a full PLM.

---

## Output Format

All scripts (except `extract_esm.py` without `--concatenate_dir`) produce a CSV file with:
- **Rows** — one per input sequence, indexed by sequence ID from the FASTA header
- **Columns** — embedding dimensions (integers 0, 1, 2, ...)

These CSVs can be loaded directly into pandas:

```python
import pandas as pd
df = pd.read_csv("embeddings.csv", index_col=0)
```

---

## Hardware Recommendations

- **GPU strongly recommended** for ESM, ProtT5, and Ankh, especially for large datasets or long sequences.
- **CPU is sufficient** for UniRep and one-hot encoding.
- Reduce `--batch_size` (Ankh, ProteinBERT) or `--toks_per_batch` (ESM) if you encounter out-of-memory errors.
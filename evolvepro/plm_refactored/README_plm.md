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

**Arguments:**
- `model_name` — Pretrained model name (e.g. `esm2_t33_650M_UR50D`, `esm1b_t33_650M_UR50S`) or path to a local `.pt` file
- `fasta_file` — Input FASTA file
- `output_dir` — Directory to save per-sequence `.pt` files
- `--include` — Representations to extract: `mean`, `per_tok`, `bos`, `contacts` (can specify multiple)
- `--repr_layers` — Layer indices to extract from (default: last layer `-1`)
- `--toks_per_batch` — Max tokens per batch (default: 4096)
- `--truncation_seq_length` — Truncate sequences longer than this (default: 1022)
- `--concatenate_dir` — If set, merges all `.pt` outputs into a single CSV

**Example:**
```bash
python extract_esm.py esm2_t33_650M_UR50D sequences.fasta esm_out/ \
    --include mean \
    --concatenate_dir results/
```

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

**Example using the recommended default:**
```bash
python extract_esm.py esm2_t33_650M_UR50D sequences.fasta esm_out/ \
    --include mean \
    --repr_layers -1 \
    --concatenate_dir results/
```

**Example using the lightweight 150M model for CPU or low-memory environments:**
```bash
python extract_esm.py esm2_t30_150M_UR50D sequences.fasta esm_out/ \
    --include mean \
    --repr_layers -1 \
    --concatenate_dir results/
```

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
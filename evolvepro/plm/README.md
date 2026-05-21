# Protein Language Model Embedding Extraction

Two scripts for extracting mean-pooled protein sequence embeddings: `extract_esmc.py` for EvolutionaryScale's ESMC models, and `extract_esm2.py` for Meta's ESM-2 models.

Both produce an identical CSV format and accept one or more FASTA files as input. Always include the wild-type sequence alongside your mutant library.

---

## Quick start

**ESMC** (use the `esm3` conda environment):
```python
from evolvepro.plm.extract_esmc import extract_embeddings

extract_embeddings(
    model='esmc_300m',
    fasta_files=['WT.fasta', 'single_mutants.fasta'],
    output_csv='embeddings_esmc_300m.csv',
    login_token='<your_huggingface_token>'
)
```

**ESM-2** (use the `esm2` conda environment):
```python
from evolvepro.plm.extract_esm2 import extract_embeddings

extract_embeddings(
    model='esm2_t33_650M_UR50D',
    fasta_files=['WT.fasta', 'single_mutants.fasta'],
    output_csv='embeddings_esm2_650m.csv'
)
```

---

## Output format

| variant | dim_0 | dim_1 | dim_2 | ... |
|---|---|---|---|---|
| WT | 0.142 | -0.037 | 0.891 | ... |
| M1A | 0.309 | 0.014 | 0.762 | ... |

One row per sequence. The `variant` column uses the FASTA header (`>name` line) as the identifier.

---

## Model options

### ESMC (`extract_esmc.py`)

| Model string | Parameters | Embedding dim | Approx. VRAM |
|---|---|---|---|
| `esmc_300m` | 300M | 960 | ~4 GB |
| `esmc_600m` | 600M | 1152 | ~8 GB |


### ESM-2 (`extract_esm2.py`)

| Model string | Parameters | Embedding dim | Approx. VRAM |
|---|---|---|---|
| `esm2_t33_650M_UR50D` | 650M | 1280 | ~5 GB |
| `esm2_t36_3B_UR50D` | 3B | 2560 | ~24 GB |

ESM-2 processes sequences in batches (`seqs_per_batch=16` by default). Reduce this if you run out of memory; all sequences must be the same length.

---

## Hardware

- A GPU is strongly recommended. The scripts will automatically use CUDA, Apple MPS, or fall back to CPU.
- If you run out of memory with ESM-2, lower `seqs_per_batch`.
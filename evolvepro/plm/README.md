# Protein Language Model Embedding Extraction


## Usage

### ESM-C (`extract_esm3.py`)

Extracts mean-pooled sequence embeddings from EvolutionaryScale's ESMC models and writes them directly to a CSV file.

```bash
python extract_esm3.py
```

The script is intended to be imported and called programmatically via `extract_embeddings()` rather than run from the command line with arguments. Configure the run by calling the function directly:

```python
from extract_esm3 import extract_embeddings

extract_embeddings(
    model="esmc_300m",
    fasta_file="sequences.fasta",
    output_csv="embeddings.csv",
    truncation_seq_length=1022,
)
```

---

### `extract_embeddings()` — Parameters

**`model`** (str, required)

The pretrained ESMC model to load. Must be a string beginning with `"esmc"` (e.g. `"esmc_300m"`, `"esmc_600m"`). ESM-3 model strings (beginning with `"esm3"`) are not supported by this script and will raise a `ValueError`.

Model weights are downloaded automatically on first use via `huggingface_hub` and cached locally.

---

**`fasta_file`** (str, required)

Path to the input FASTA file containing the protein sequences to embed. Each sequence header (the `>name` line) is used as the row label in the output CSV.

---

**`output_csv`** (str, required)

Path for the output CSV file. If the file already exists, the script exits without overwriting it. The CSV is written incrementally — each sequence is flushed to disk immediately after processing, so results are preserved even if the run is interrupted.

---

**`truncation_seq_length`** (int, default: `1022`)

Sequences longer than this value are truncated before embedding. Residues beyond the limit are silently dropped. The default of 1022 matches the positional encoding limit of ESM-2-era models; adjust upward if your ESMC model supports longer contexts and your sequences require it.

---

**`login_token`** (str, default: `None`)

Optional Hugging Face login token for accessing gated model repositories. Pass your token here if model download fails with a `GatedRepoError`. You can also authenticate in advance using `huggingface-cli login`.

---

### Device selection

The script automatically selects the best available device in order of preference: CUDA GPU → Apple MPS → CPU. No manual configuration is needed. A message is printed at startup confirming which device is in use.

---

### Output format

The script produces a single CSV file with:

- **Rows** — one per input sequence, labelled by sequence ID from the FASTA header
- **Columns** — `variant` (the sequence ID), followed by embedding dimensions `dim_0`, `dim_1`, ..., `dim_N`

Embeddings are mean-pooled across all residue positions (BOS and EOS tokens are excluded before pooling), producing one fixed-length vector per sequence regardless of sequence length.

**Example output (truncated):**

| variant | dim_0 | dim_1 | dim_2 | ... |
|---|---|---|---|---|
| seq_A | 0.142 | -0.037 | 0.891 | ... |
| seq_B | 0.309 | 0.014 | 0.762 | ... |

---

### ESMC Model Variants

ESMC is available in two sizes. Both are accessed via the `esm` package from EvolutionaryScale and share the same API in this script.

| Model string | Parameters | Embedding dim | Approx. VRAM |
|---|---|---|---|
| `esmc_300m` | 300M | 960 | ~4 GB |
| `esmc_600m` | 600M | 1152 | ~8 GB |

**Choosing a model:**

- `esmc_300m` is a practical default for most protein engineering tasks, offering strong embedding quality with moderate memory requirements. It runs comfortably on a single consumer GPU.
- `esmc_600m` provides richer representations and is worth using when downstream task performance is the priority and the hardware budget allows it.
- Both models support sequences up to `truncation_seq_length` residues. Neither has an inherent advantage for longer sequences.

---

### Comparison with the legacy script (`extract_esm_legacy.py`)

The legacy script targets Meta's ESM-2 family via the `fair-esm` package and uses a two-step pipeline: it first writes per-sequence `.pt` files, then optionally concatenates them into a CSV. The current script targets EvolutionaryScale's ESMC family via the newer `esm` package and writes directly to CSV in a single pass, with no intermediate files.

| | `extract_esm_legacy.py` | `extract_esm3.py` |
|---|---|---|
| Model family | ESM-2 (Meta, `fair-esm`) | ESMC (EvolutionaryScale, `esm`) |
| Intermediate files | Per-sequence `.pt` files | None |
| Output | CSV (via concatenation step) | CSV (written directly) |
| Representation types | `mean`, `per_tok`, `bos`, `contacts` | `mean` only |
| Multiple layers | Supported | Not applicable |
| CLI interface | Yes (`argparse`) | No (call `extract_embeddings()`) |

Use the legacy script if you need ESM-2 models, per-residue embeddings, contact map extraction, or multi-layer representations. Use this script for ESMC models with mean-pooled embeddings.

---

### Hardware recommendations

- A GPU is strongly recommended for large datasets or the 600M model.
- The 300M model can be run on CPU for small datasets, though it will be slow.
- If you encounter out-of-memory errors, reduce the number of sequences processed concurrently by splitting your FASTA file into smaller batches and running the script once per batch (the output file check prevents overwriting).
#!/usr/bin/env python3 -u

# Updated to incorporate the new esm package which can access ESM-C

import torch
import csv
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from evolvepro.plm.helpers import *


def load_model(model: str, device: torch.device, login_token: str = None):
    """Load either an ESMC or ESM3 model depending on the model string."""

    if model.startswith("esm3"):
        raise ValueError("ESM3 models are not supported (requires paid API or special access). Please use ESMC instead.")
    elif model.startswith("esmc"):
        client = ESMC.from_pretrained(model).to(device)
    else:
        raise ValueError(f"Unknown model: {model}")

    client.eval()
    return client


def get_embeddings(client, seq: str):
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(protein_tensor,
                                  LogitsConfig(sequence=True, return_embeddings=True))
    embeddings = logits_output.embeddings[0]    # (L+2, D), includes BOS/EOS
    embeddings = embeddings[1:-1]               # remove BOS and EOS
    embeddings = embeddings.mean(0)             # mean-pooling over residues
    embeddings = embeddings.clone()             # makes a copy so it can be cleared from GPU
    embeddings = embeddings.cpu()               # moves to CPU for writing
    return embeddings


def extract_embeddings(model: str,
                       fasta_files: str | list[str],
                       output_csv: str,
                       truncation_seq_length: int = 1022,
                       login_token: str = None):

    device = get_device()
    print(f"Using device: {device}")

    client = load_model(model=model, device=device, login_token=login_token)
    print(f"Loaded model: {model}")

    if isinstance(fasta_files, str):
        fasta_files = [fasta_files]

    sequences = []
    for fasta_file in fasta_files:
        seqs = parse_fasta(fasta_file)
        sequences.extend(seqs)
        print(f"Read {fasta_file} with {len(seqs)} sequences")

    if pathlib.Path(output_csv).exists():
        print(f"Output file {output_csv} already exists. Exiting.")
        return

    with torch.no_grad(), open(output_csv, "w") as out:
        writer = csv.writer(out)
        header_written = False  # Flag to ensure header is written only once

        for i, (label, seq) in enumerate(sequences):
            print(f"Processing sequence {i + 1} of {len(sequences)}: {label}")

            seq = seq[:truncation_seq_length]   # Truncate if needed
            embeddings = get_embeddings(client=client, seq=seq)

            if not header_written:
                writer.writerow(["variant"] + [f"dim_{j}" for j in range(len(embeddings))])
                header_written = True

            writer.writerow([label] + embeddings.tolist())
            out.flush()   # write to disk immediately

    print(f"Embeddings saved to {output_csv}.")

#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This is meant to be used with the fair-esm package.
# ESM-C needs to be accessed using the new esm package

import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
import re
import csv
import pathlib
import torch
from Bio import SeqIO
from evolvepro.plm.helpers import *

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


def load_model(model: str, device: torch.device):
    """Loads the model"""
    esm_model, alphabet = pretrained.load_model_and_alphabet(model)
    esm_model = esm_model.to(device)
    esm_model.eval()
    return esm_model, alphabet


def get_embeddings(client, seq: str):
    esm_model, alphabet = client
    device = next(esm_model.parameters()).device

    # Tokenize: batch_converter expects a list of (label, sequence) tuples
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("seq", seq)])
    tokens = tokens.to(device)

    # Run the forward pass, requesting the last transformer layer's output.
    # num_layers tells us which layer index to ask for.
    num_layers = esm_model.num_layers
    result = esm_model(tokens, repr_layers=[num_layers])

    # result["representations"][num_layers] has shape (1, L+2, D):
    #   - 1        : batch size (we process one sequence at a time)
    #   - L+2      : sequence length + BOS and EOS special tokens
    #   - D        : embedding dimension (e.g. 480 for ESM2-35M, 1280 for ESM2-650M)
    token_embeddings = result["representations"][num_layers]

    # Strip the BOS (index 0) and EOS (index -1) tokens, then mean-pool
    embeddings = token_embeddings[0, 1:-1].mean(0)
    embeddings = embeddings.clone().cpu()
    return embeddings


def get_batch_embeddings(client, labels: list[str], seqs: list[str]):
    """Run a single batched forward pass and return mean-pooled embeddings for each sequence.

    Args:
        client:  (model, alphabet) tuple from load_model()
        labels:  list of sequence identifiers (variant names)
        seqs:    list of amino acid sequence strings, all the same length

    Returns:
        list of (label, embedding) tuples, one per sequence in the batch
    """
    esm_model, alphabet = client
    device = next(esm_model.parameters()).device
    num_layers = esm_model.num_layers

    # Tokenize each sequence individually, then stack into a single batch tensor.
    # Because all sequences are the same length, the resulting tokens tensors are
    # all the same shape and can be stacked directly without any padding.
    batch_converter = alphabet.get_batch_converter()
    token_list = []
    for label, seq in zip(labels, seqs):
        _, _, tokens = batch_converter([(label, seq)])
        token_list.append(tokens[0])  # tokens[0]: shape (L+2,) — one sequence
    tokens = torch.stack(token_list).to(device)  # shape: (batch_size, L+2)

    result = esm_model(tokens, repr_layers=[num_layers])

    # representations shape: (batch_size, L+2, D)
    # Move to CPU and float32 immediately to free device memory
    representations = result["representations"][num_layers].to(device="cpu", dtype=torch.float32)

    # Strip BOS (index 0) and EOS (index -1), then mean-pool across residues.
    # No per-sequence length tracking needed since all sequences are the same length.
    embeddings = representations[:, 1:-1, :].mean(dim=1)  # shape: (batch_size, D)

    return list(zip(labels, embeddings.unbind(dim=0)))


def extract_embeddings(model: str,
                       fasta_files: str | list[str],
                       output_csv: str,
                       truncation_seq_length: int = 1022,
                       seqs_per_batch: int = 16):
    """Extract mean-pooled ESM embeddings for all sequences in one or more FASTA files.

    All sequences must be the same length (e.g. single-mutant libraries of a fixed
    protein). Sequences are processed in fixed-size batches for hardware efficiency.

    Args:
        model:                  fair-esm model name (e.g. 'esm2_t36_3B_UR50D')
        fasta_files:            path or list of paths to FASTA files
        output_csv:             path for the output CSV file
        truncation_seq_length:  sequences longer than this are truncated before embedding
        seqs_per_batch:         number of sequences per forward pass; reduce if you run
                                out of memory, increase if hardware is underutilized
    """
    device = get_device()
    print(f"Using device: {device}")

    client = load_model(model=model, device=device)
    print(f"Loaded model: {model}")

    if isinstance(fasta_files, str):
        fasta_files = [fasta_files]

    sequences = []
    for fasta_file in fasta_files:
        seqs = parse_fasta(fasta_file)
        sequences.extend(seqs)
        print(f"Read {fasta_file} with {len(seqs)} sequences")

    # Truncate and validate that all sequences are the same length
    sequences = [(label, seq[:truncation_seq_length]) for label, seq in sequences]
    seq_lengths = set(len(seq) for _, seq in sequences)
    if len(seq_lengths) > 1:
        raise ValueError(f"All sequences must be the same length, but found lengths: {seq_lengths}")

    if pathlib.Path(output_csv).exists():
        print(f"Output file {output_csv} already exists. Exiting.")
        return

    # Split into fixed-size batches
    batches = [sequences[i:i + seqs_per_batch] for i in range(0, len(sequences), seqs_per_batch)]
    print(f"Processing {len(sequences)} sequences in {len(batches)} batches of up to {seqs_per_batch}")

    with torch.no_grad(), open(output_csv, "w") as out:
        writer = csv.writer(out)
        header_written = False
        sequences_processed = 0

        for batch_idx, batch in enumerate(batches):
            batch_labels = [label for label, _ in batch]
            batch_seqs = [seq for _, seq in batch]
            print(f"Processing batch {batch_idx + 1} of {len(batches)} ({len(batch_labels)} sequences)")

            batch_results = get_batch_embeddings(client=client, labels=batch_labels, seqs=batch_seqs)

            for label, embedding in batch_results:
                sequences_processed += 1
                print(f"  Writing sequence {sequences_processed} of {len(sequences)}: {label}")

                if not header_written:
                    writer.writerow(["variant"] + [f"dim_{j}" for j in range(len(embedding))])
                    header_written = True

                writer.writerow([label] + embedding.tolist())

            out.flush()

    print(f"Embeddings saved to {output_csv}.")

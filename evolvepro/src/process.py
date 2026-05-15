import os
from itertools import combinations
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



def generate_wt(wt_sequence: str, output_file: str) -> None:
    """Write a single-record FASTA file containing the wild-type sequence.

    The resulting file is used throughout the pipeline wherever ``wt_fasta_path``
    is required.

    Args:
        wt_sequence: Full wild-type protein sequence (one-letter amino acid codes).
        output_file: Destination path for the FASTA file, e.g. ``"WT.fasta"``.
    """
    record = SeqRecord(Seq(wt_sequence), id='WT', description='')
    with open(output_file, 'w') as fh:
        SeqIO.write(record, fh, 'fasta')


def generate_single_aa_mutants(
    wt_fasta: str,
    output_file: str,
    positions: Optional[List[int]] = None,
) -> None:
    """Generate all possible single amino-acid substitutions and save as FASTA.

    Iterates over every position (or a specified subset) and every amino acid
    in the standard 20-letter alphabet, creating one FASTA record per unique
    substitution.  The WT sequence is included as the first record.

    Variant names follow the standard convention: ``<WT AA><position><mutant AA>``
    (1-based), e.g. ``"A107M"``.

    This FASTA is the starting point for generating protein embeddings.

    Args:
        wt_fasta: Path to the wild-type FASTA file.
        output_file: Destination path for the mutant FASTA, e.g. ``"mutants.fasta"``.
        positions: 1-based list of positions to mutate.  ``None`` mutates all
            positions in the sequence.
    """
    aa_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    wt_sequence = SeqIO.read(wt_fasta, 'fasta').seq
    records = [SeqRecord(wt_sequence, id='WT', description='')]

    positions_to_mutate = (
        [p - 1 for p in positions]          # convert to 0-based
        if positions is not None
        else range(len(wt_sequence))
    )

    for i in positions_to_mutate:
        wt_aa = wt_sequence[i]
        for mut_aa in aa_alphabet:
            if mut_aa == wt_aa:
                continue
            mutant_seq = wt_sequence[:i] + mut_aa + wt_sequence[i + 1:]
            variant = f'{wt_aa}{i + 1}{mut_aa}'
            records.append(SeqRecord(Seq(mutant_seq), id=variant, description=''))

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as fh:
        SeqIO.write(records, fh, 'fasta')

    print(f'Mutant FASTA written: {len(records)} sequences → {output_file}')


def generate_n_mutant_combinations(
    wt_fasta: str,
    mutant_file: str,
    n: int,
    output_file: str,
    threshold: float = 1.0,
) -> None:
    """Generate all n-fold combinations of beneficial single mutants.

    Reads a list of single mutants from an Excel file, filters to those above a
    minimum activity threshold, and writes a FASTA containing every valid
    n-mutation combination (combinations that mutate the same position more than
    once are skipped).

    Multi-mutant variant names join single-mutant tokens with underscores:
    ``"A107M_F73C"``.

    Args:
        wt_fasta: Path to the wild-type FASTA file.
        mutant_file: Path to an Excel file with ``Variant`` and ``activity``
            columns.  ``Variant`` values should be in position-only format
            (e.g. ``"107M"``).  Only variants with
            ``activity > threshold`` are included.
        n: Number of mutations to combine (e.g. ``2`` for double mutants).
        output_file: Destination path for the output FASTA.
        threshold: Minimum single-mutant activity score for inclusion.
    """
    wt_sequence = str(SeqIO.read(wt_fasta, 'fasta').seq)

    mutants = pd.read_excel(mutant_file)
    mutants = mutants[mutants['activity'] > threshold].copy()

    # Convert position-only notation to standard notation.
    mutants[['position', 'mutant_aa']] = (
        mutants['Variant'].str.extract(r'(\d+)([A-Z]+)', expand=True)
    )
    mutants['wt_aa'] = mutants['position'].apply(
        lambda p: wt_sequence[int(p) - 1]
    )
    mutants['variant'] = mutants['wt_aa'] + mutants['position'] + mutants['mutant_aa']

    records = [SeqRecord(Seq(wt_sequence), id='WT', description='Wild-type sequence')]
    combos = list(combinations(mutants['variant'], n))
    valid_count = 0

    for combo in combos:
        positions_used: set = set()
        valid = True
        mutant_seq = wt_sequence
        name_parts = []

        for mutant in combo:
            wt_aa, pos_str, mut_aa = mutant[0], mutant[1:-1], mutant[-1]
            pos_idx = int(pos_str) - 1          # 0-based index

            if pos_idx in positions_used:       # conflicting mutations → skip
                valid = False
                break

            positions_used.add(pos_idx)
            mutant_seq = mutant_seq[:pos_idx] + mut_aa + mutant_seq[pos_idx + 1:]
            name_parts.append(mutant)

        if valid:
            records.append(
                SeqRecord(Seq(mutant_seq), id='_'.join(name_parts), description='')
            )
            valid_count += 1

    print(f'Combinations generated:  {len(combos)}')
    print(f'Valid combinations written: {valid_count}')

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as fh:
        SeqIO.write(records, fh, 'fasta')
